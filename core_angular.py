import torch
import numpy as np
from torch.optim import lr_scheduler
from torch import nn
import cv2
import os
import random
from field_propagation import Field_propagation

# torch.set_default_dtype(torch.float64)

class Core(Field_propagation):
    def __init__(self, flags):
        self.MS_property = []
        self.flags = flags  # The Flags containing the specs
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device=device

        if len(self.flags.angle_x) != len(self.flags.angle_y):
            raise ValueError("angle_x and angle_y must have the same length.")
        self.case_configs = list(zip(self.flags.angle_x, self.flags.angle_y))



    def make_optimizer_eval(self, optimizer_type=None):
        """
        The function to make the optimizer during evaluation time.
        The difference between optm is that it does not have regularization and it only optmize the self.geometr_eval tensor
        :return: the optimizer_eval
        """
        if optimizer_type is None:
            optimizer_type = self.flags.optim
        if optimizer_type == 'Adam':
            op = torch.optim.Adam(self.MS_property, lr=self.flags.lr)
        elif optimizer_type == 'RMSprop':
            op = torch.optim.RMSprop(self.MS_property, lr=self.flags.lr)
        elif optimizer_type == 'SGD':
            op = torch.optim.SGD(self.MS_property, lr=self.flags.lr)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def make_lr_scheduler(self, optm):

        return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min', factor=self.flags.lr_decay_rate,
                                              patience=100, threshold=1e-2)


    def model(self, micro_batch_configs):
        ms_value = self.MS_property
        Num_layer = len(ms_value)
        batch_size = len(micro_batch_configs)

        Inc, wavelength = self.angle_phase(micro_batch_configs)
        J_total = Inc

        # First metasurface layer and gap propagation.
        MS_0 = torch.exp(1j * 2 * np.pi * ms_value[0])
        MS_0 = MS_0.repeat(batch_size, 1, 1)
        J_total_inter = MS_0 * J_total

        J_total = self.func(
            J_total_inter,
            wavelength,
            self.flags.distance[0],
            self.flags.refractive_index[0],
        )

        # Second metasurface layer and propagation to the hologram plane.
        if Num_layer > 1:
            MS_1 = torch.exp(1j * 2 * np.pi * ms_value[1])
            MS_1 = MS_1.repeat(batch_size, 1, 1)
            J_total_inter = MS_1 * J_total
            J_total = self.func(J_total_inter, wavelength, self.flags.distance[1], self.flags.refractive_index[1])

        return J_total


    def XY_generation(self):

        x = self.flags.x
        y = self.flags.y
        Nx = self.flags.Nx
        Ny = self.flags.Ny

        Lx = x[-1] - x[0]
        Ly = y[-1] - y[0]

        sj = torch.linspace(-Lx / 2, Lx / 2, Nx + 1, device=self.device)  # this is much faster
        sj = sj[:-1]
        nj = torch.linspace(-Ly / 2, Ly / 2, Ny + 1, device=self.device)
        nj = nj[:-1]
        X, Y = torch.meshgrid(sj, nj, indexing='ij')

        return X,Y


    def angle_phase(self, micro_batch_configs):
        batch_size = len(micro_batch_configs)
        Num_wave=len(self.flags.wavelength)
        # Extract x and y incidence angles.
        angles_x_in_batch = [config[0] for config in micro_batch_configs]
        angles_y_in_batch = [config[1] for config in micro_batch_configs]

        # Generate the incidence-angle phase.
        wavelength_tensor = self.build_tensor(self.flags.wavelength)[:, None, None]
        k = 2 * np.pi / wavelength_tensor

        angles_x_rad = np.pi / 180 * self.build_tensor(angles_x_in_batch)[:, None, None]
        angles_y_rad = np.pi / 180 * self.build_tensor(angles_y_in_batch)[:, None, None]

        X,Y=self.XY_generation()


        # Use broadcasting over cases and wavelengths.
        # k: (Num_wave*batch_size, 1, 1)
        # angles_x_rad: (Num_wave*batch_size, 1, 1, 1)
        # angles_y_rad: (Num_wave*batch_size, 1, 1, 1)
        k_b = k.repeat(batch_size,1,1)
        angles_x_rad_b = torch.repeat_interleave(angles_x_rad,Num_wave,dim=0)
        angles_y_rad_b = torch.repeat_interleave(angles_y_rad,Num_wave,dim=0)

        # X, Y: (Nx, Ny)
        # The incident phase now includes both x and y angle components.
        incident_field = torch.exp(
            1j * k_b * (torch.sin(angles_x_rad_b) * X + torch.sin(angles_y_rad_b) * Y)
        )

        wavelength = self.flags.wavelength * batch_size

        return incident_field, wavelength



    def make_loss(self, logit, labels):
        return nn.MSELoss()(logit, self.flags.eta * labels)

    def read_figure(self):

        data_dir = self.flags.data_dir

        # Compute the target region from ratio and the observation-grid dimensions.
        apsizex = self.flags.x[-1] - self.flags.x[0]
        apsizey = self.flags.y[-1] - self.flags.y[0]
        obsizex = self.flags.x_ob[-1] - self.flags.x_ob[0]
        obsizey = self.flags.y_ob[-1] - self.flags.y_ob[0]

        ratio_x = obsizex / apsizex
        ratio_y = obsizey / apsizey

        Nx_ob = int(self.flags.Nx * ratio_x)
        Ny_ob = int(self.flags.Ny * ratio_y)

        ratio = self.flags.ratio

        # Size of the active image region.
        nxx = int(Nx_ob * ratio)
        nyy = int(Ny_ob * ratio)

        # Starting coordinates of the active image region.
        x1 = int((1 - ratio) / 2 * Nx_ob)
        y1 = int((1 - ratio) / 2 * Ny_ob)

        Num_wave = len(self.flags.wavelength)
        total_cases = Num_wave * len(self.case_configs)


        images = []

        for i in range(total_cases):
            # Read the target image for this case.
            filename = os.path.join(data_dir, "{}.bmp".format(i))

            if not os.path.exists(filename):
                # Fall back to 0.bmp when the requested numbered image is unavailable.
                # print(f"Warning: {filename} not found, repeating 0.bmp")
                filename = os.path.join(data_dir, "0.bmp")

            image = cv2.imread(filename)

            # Preserve the preprocessing used for the archived calculations.
            image = image[:, :, 0]
            image = np.rot90(image, -1)
            image = cv2.resize(image, (nyy, nxx))

            max_value = np.max(image)
            if max_value > 0:
                image = image / max_value

            image0 = np.zeros([Nx_ob, Ny_ob])
            image0[x1:x1 + nxx, y1:y1 + nyy] = image
            images.append(image0)

        result = self.build_tensor(np.stack(images, axis=0))
        return result

    def ratio_image(self,image):
        apsizex = self.flags.x[-1] - self.flags.x[0]
        apsizey = self.flags.y[-1] - self.flags.y[0]
        obsizex = self.flags.x_ob[-1] - self.flags.x_ob[0]
        obsizey = self.flags.y_ob[-1] - self.flags.y_ob[0]

        ratio_x = obsizex / apsizex
        ratio_y = obsizey / apsizey

        Nx_ob = int(self.flags.Nx * ratio_x)
        Ny_ob = int(self.flags.Ny * ratio_y)

        ratio = self.flags.ratio

        # Size of the active image region.
        nxx = int(Nx_ob * ratio)
        nyy = int(Ny_ob * ratio)

        # Starting coordinates of the active image region.
        x1 = int((1 - ratio) / 2 * Nx_ob)
        y1 = int((1 - ratio) / 2 * Ny_ob)

        image_t=image[:,x1:x1 + nxx, y1:y1 + nyy]
        return image_t


    def train(self):

        # Initialize the geometry_eval or the initial guess xs
        self.initialize_metasurface()
        # Set up the learning schedule and optimizer
        self.optm_eval = self.make_optimizer_eval()  # , optimizer_type='SGD')
        self.lr_scheduler = self.make_lr_scheduler(self.optm_eval)
        labels = self.read_figure()

        self.iteration = []
        self.save_loss = []

        # Split the label tensor by case, preserving the order in self.case_configs.
        labels_per_case = torch.chunk(labels, len(self.case_configs), dim=0)
        num_micro_batches = (
            len(self.case_configs) + self.flags.batch_size - 1
        ) // self.flags.batch_size

        for epoch in range(self.flags.train_step):
            self.optm_eval.zero_grad()
            total_loss_value = 0.0

            # Gradient-accumulation loop; each iteration processes one micro-batch.
            for i in range(num_micro_batches):
                # Slice configurations and labels for the current micro-batch.
                start = i * self.flags.batch_size
                end = start + self.flags.batch_size

                micro_batch_configs = self.case_configs[start:end]
                if not micro_batch_configs: continue  # Skip an empty final micro-batch.

                micro_batch_labels_list = labels_per_case[start:end]
                micro_batch_labels = torch.cat(micro_batch_labels_list, dim=0)

                # Forward propagation.
                logit = self.model(micro_batch_configs)
                logit = torch.abs(logit) ** 2

                # Compute the loss.
                loss = self.make_loss(logit, micro_batch_labels) / num_micro_batches

                # Backpropagate the accumulated loss.
                loss.backward()
                total_loss_value += loss.item()

            self.optm_eval.step()  # Move one step the optimizer
            self.lr_scheduler.step(total_loss_value)
            if not epoch % 10:
                print("This is Epoch {:4d}, training loss {:.5f}".format(epoch, total_loss_value))

            self.save_loss.append(total_loss_value)
            self.iteration.append(epoch)

            self.lr_scheduler.step(total_loss_value)

        print("Training finished. Evaluating the final model to calculate crosstalk...")
        # self.model.eval()  # The model contains no stochastic train/eval layers.
        all_logits = []  # Collect outputs from every micro-batch.

        # Disable gradient tracking during final evaluation.
        with torch.no_grad():
            # Determine the number of micro-batches needed to cover every case.
            for i in range(num_micro_batches):
                # Slice the current micro-batch configurations.
                start = i * self.flags.batch_size
                end = start + self.flags.batch_size

                micro_batch_configs = self.case_configs[start:end]
                if not micro_batch_configs:
                    continue  # Skip an empty final micro-batch.

                # Forward propagation.
                logit_batch = self.model(micro_batch_configs)
                logit_batch = torch.abs(logit_batch) ** 2

                # Append the current batch output.
                all_logits.append(logit_batch)

        # Concatenate all micro-batch outputs along the batch dimension.
        full_logit = torch.cat(all_logits, dim=0)

        # Verify that calculated fields and target labels have identical shapes.
        assert full_logit.shape == labels.shape, \
            f"Shape mismatch! full_logit: {full_logit.shape}, labels: {labels.shape}"

        # Calculate crosstalk from the full output and label tensors.
        self.crosstalk, self.crosstalk_average = self.calculate_confusion_matrix_torch(self.ratio_image(labels), self.ratio_image(full_logit))

        # self.crosstalk_diff, self.crosstalk_average_diff = self.calculate_crosstalk_matrix_torch_diff(labels, full_logit)

    def initialize_metasurface(self):

        np.random.seed(13)
        random.seed(13)
        Number_ms = len(self.flags.refractive_index)

        output = []
        for i in range(0, Number_ms):
            phy = torch.full([1,self.flags.Nx, self.flags.Ny], random.random())
            phy=self.build_tensor(phy,requires_grad=True)
            layer = [phy]
            output += layer
        self.MS_property = output

    def calculate_confusion_matrix_torch(self,design_images, measured_images, threshold=0.3):
        """
        Calculate an energy-integral confusion matrix with PyTorch.

        Definition of M_ij:
            M[i, j] is the fraction of reconstructed-channel-i energy that falls
            inside target region j.

            Denominator: total energy in reconstructed image R_i.
            Numerator:
                - If i == j, integrate over the signal region T_i == 1.
                - If i != j, integrate over the ghost region T_i == 0 and T_j == 1.

        Args:
            design_images (torch.Tensor):
                Ideal target images T with shape (N, H, W).
            measured_images (torch.Tensor):
                Reconstructed images R with shape (N, H, W).
            threshold (float):
                Threshold used to binarize target images; default is 0.3.

        Returns:
            torch.Tensor: N x N confusion matrix M.
            float: mean crosstalk, defined as the off-diagonal average.
        """
        # Validate inputs.
        if design_images.dim() != 3 or measured_images.dim() != 3:
            raise ValueError("Input tensors must be 3D with shape (N, H, W).")
        if design_images.shape != measured_images.shape:
            raise ValueError("design_images and measured_images must have identical shapes.")

        N = design_images.shape[0]
        device = design_images.device
        dtype = measured_images.dtype
        epsilon = 1e-12

        if N == 0:
            return torch.empty((0, 0), device=device, dtype=dtype), 0.0

        # Build binary target masks T.
        T_bool = design_images > threshold

        # Denominator: total energy in each reconstructed image.
        total_energy = measured_images.sum(dim=(1, 2))  # Shape: (N,)

        # Calculate matrix elements.
        confusion_matrix = torch.zeros((N, N), device=device, dtype=dtype)

        # Fill one reconstructed-image row M[i, j] at a time.
        for i in range(N):
            current_R = measured_images[i]  # R_i: (H, W)
            current_T_i_bool = T_bool[i]  # T_i: (H, W)
            current_not_T_i_bool = ~current_T_i_bool  # Dark region outside T_i.

            # Construct the integration region for every target index j in row i.

            # Broadcast ROI masks to shape (N, H, W).
            # ROI[j] =
            #   If i == j: T_i (Signal Region)
            #   If i != j: T_i_inv AND T_j (Ghost Region)

            # Start with each ghost region: T_j AND (NOT T_i).
            # (N, H, W) & (H, W) -> (N, H, W)
            roi_masks_bool = T_bool & current_not_T_i_bool

            # Replace the diagonal region with the signal region T_i.
            roi_masks_bool[i] = current_T_i_bool

            # Convert masks to the measured-image dtype for multiplication.
            roi_masks = roi_masks_bool.to(dtype=dtype)

            # Numerator: integrated energy inside each ROI.
            # current_R (H, W) * roi_masks (N, H, W) -> (N, H, W) -> Sum over H,W -> (N,)
            energy_in_roi = (current_R * roi_masks).sum(dim=(1, 2))

            # Normalize every element by the total energy in row i.
            # energy_in_roi (N,) / scalar
            confusion_matrix[i, :] = energy_in_roi / (total_energy[i] + epsilon)

        # Mean crosstalk is the average of off-diagonal elements only.

        # Select off-diagonal elements.
        eye_mask = torch.eye(N, device=device).bool()
        off_diagonal_elements = confusion_matrix[~eye_mask]  # Flattened off-diagonal

        if off_diagonal_elements.numel() > 0:
            average_crosstalk = off_diagonal_elements.mean()
        else:
            average_crosstalk = 0.0

        return confusion_matrix, average_crosstalk
