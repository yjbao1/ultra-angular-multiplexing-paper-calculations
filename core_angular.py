import torch
import numpy as np
from torch.fft import fft2, fftshift, ifft2
from torch.nn.functional import relu
from torch.optim import lr_scheduler
from torch import nn
import torch.nn.functional as F
import time
import cv2
import os
from PIL import Image
import random
import itertools
from field_propagation import Field_propagation

# torch.set_default_dtype(torch.float64)

class Core(Field_propagation):
    def __init__(self, flags):
        self.MS_property = []
        self.flags = flags  # The Flags containing the specs
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device=device

        # Pair the x and y incidence angles.
        # zip groups corresponding entries from the two angle lists.
        # e.g., [(-20, -20), (0, 0), (20, 20)]
        angle_pairs = list(zip(self.flags.Angle_x, self.flags.Angle_y))

        # Read the topological-charge list.
        topos = self.flags.topology_vortex

        # Form the Cartesian product of angle pairs and topological charges.
        #    e.g., [ ((-20, -20), -6), ((-20, -20), -3), ..., ((20, 20), 6) ]
        self.case_configs = list(itertools.product(angle_pairs, topos))

        # Flatten every configuration to (angle_x, angle_y, topological_charge).
        self.case_configs = [
            (angle_pair[0], angle_pair[1], topo)
            for angle_pair, topo in self.case_configs
        ]



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


    def rotate_complex_field(self, field, angle_deg):
        """
        Rotate a complex field about its physical center using differentiable PyTorch interpolation.
        field: complex tensor with shape (Batch, Nx, Ny) or (Batch, C, Nx, Ny)
        angle_deg: rotation angle in degrees
        """
        original_shape = field.shape
        # Reshape to the four-dimensional (N, C, H, W) layout required by F.grid_sample.
        # Flatten all leading dimensions into the batch dimension.
        field_4d = field.view(-1, 1, field.shape[-2], field.shape[-1])
        N, C, H, W = field_4d.shape

        angle_rad = angle_deg * np.pi / 180.0
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Build the 2 x 3 affine transformation matrix.
        # [cos(a), -sin(a), 0]
        # [sin(a),  cos(a), 0]
        theta_mat = torch.tensor([
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0]
        ], dtype=torch.float32, device=field.device)

        # Expand the transform over the batch dimension.
        theta_mat = theta_mat.unsqueeze(0).repeat(N, 1, 1)

        # Generate the sampling grid.
        grid = F.affine_grid(theta_mat, size=(N, C, H, W), align_corners=False)

        # Interpolate real and imaginary parts separately for compatibility and stability.
        # padding_mode='zeros' applies an absorbing boundary outside the field.
        field_real = field_4d.real.float()
        field_imag = field_4d.imag.float()

        out_real = F.grid_sample(field_real, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        out_imag = F.grid_sample(field_imag, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Recombine the complex field and restore the original shape.
        out_complex = torch.complex(out_real, out_imag).to(field.dtype)
        return out_complex.view(original_shape)

    def model(self, micro_batch_configs, apply_random_rotation=False, specific_rotation=0.0,
              apply_random_distance_error=False, apply_random_angle_error=False,
              rotation_error_mode='batch', distance_error_mode='batch', angle_error_mode='sample'):
        # Rotation, distance, and angle errors can be shared by a batch or sampled independently.

        ms_value = self.MS_property
        Num_layer = len(ms_value)
        Num_wave = len(self.flags.wavelength)
        batch_size = len(micro_batch_configs)

        # Incidence-angle error: shared by a batch or sampled independently.
        if apply_random_angle_error and hasattr(self.flags, 'max_angle_error') and self.flags.max_angle_error > 0:
            if angle_error_mode == 'batch':
                angle_noise_x = (torch.rand(1, device=self.device).item() * 2 - 1) * self.flags.max_angle_error
                angle_noise_y = (torch.rand(1, device=self.device).item() * 2 - 1) * self.flags.max_angle_error
                perturbed_configs = [(cfg[0] + angle_noise_x, cfg[1] + angle_noise_y, cfg[2]) for cfg in micro_batch_configs]
            else:
                angle_noise_x = (torch.rand(batch_size, device=self.device) * 2 - 1) * self.flags.max_angle_error
                angle_noise_y = (torch.rand(batch_size, device=self.device) * 2 - 1) * self.flags.max_angle_error
                perturbed_configs = []
                for cfg, dx, dy in zip(micro_batch_configs, angle_noise_x.tolist(), angle_noise_y.tolist()):
                    perturbed_configs.append((cfg[0] + dx, cfg[1] + dy, cfg[2]))
        else:
            perturbed_configs = micro_batch_configs

        Inc, wavelength = self.Angle_phase_multiple_vortex(perturbed_configs)
        J_total = Inc

        # First metasurface layer and gap propagation.
        MS_0 = torch.exp(1j * 2 * np.pi * ms_value[0])
        MS_0 = MS_0.repeat(batch_size, 1, 1)
        J_total_inter = MS_0 * J_total

        distance_0 = self.flags.distance[0]
        if apply_random_distance_error and hasattr(self.flags, 'max_distance_error') and self.flags.max_distance_error > 0:
            if distance_error_mode == 'batch':
                distance_0 = distance_0 + (torch.rand(1).item() * 2 - 1) * self.flags.max_distance_error
            else:
                distance_0 = distance_0 + (torch.rand(batch_size, device=self.device) * 2 - 1) * self.flags.max_distance_error

        J_total = self.func(J_total_inter, wavelength, distance_0, self.flags.refractive_index[0])

        self.field_on_second_layer = J_total
        # Field rotation.
        # Apply a random robustness rotation during training when requested.
        if apply_random_rotation and hasattr(self.flags, 'max_rotation_angle'):
            if rotation_error_mode == 'batch':
                # Draw a uniform random angle from [-max_angle, max_angle].
                theta = (torch.rand(1).item() * 2 - 1) * self.flags.max_rotation_angle
                if theta != 0.0:
                    J_total = self.rotate_complex_field(J_total, theta)
            else:
                theta = (torch.rand(batch_size, device=self.device) * 2 - 1) * self.flags.max_rotation_angle
                J_total_list = []
                for idx in range(batch_size):
                    theta_i = theta[idx].item()
                    J_i = J_total[idx:idx + 1]
                    if theta_i != 0.0:
                        J_i = self.rotate_complex_field(J_i, theta_i)
                    J_total_list.append(J_i)
                J_total = torch.cat(J_total_list, dim=0)
        else:
            # Use the specified angle during evaluation or testing.
            theta = specific_rotation
            # Rotate the field before the second layer when the requested angle is nonzero.
            if theta != 0.0:
                J_total = self.rotate_complex_field(J_total, theta)

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


    def Angle_phase_multiple_vortex(self, micro_batch_configs):
        # micro_batch_configs: [ (ax1, ay1, t1), (ax2, ay2, t2), ... ]

        batch_size = len(micro_batch_configs)
        Num_wave=len(self.flags.wavelength)
        # Extract x angles, y angles, and topological charges.
        angles_x_in_batch = [config[0] for config in micro_batch_configs]
        angles_y_in_batch = [config[1] for config in micro_batch_configs]
        topos_in_batch = [config[2] for config in micro_batch_configs]

        # Generate the incidence-angle phase.
        wavelength_tensor = self.build_tensor(self.flags.wavelength)[:, None, None]
        k = 2 * np.pi / wavelength_tensor

        angles_x_rad = np.pi / 180 * self.build_tensor(angles_x_in_batch)[:, None, None]
        angles_y_rad = np.pi / 180 * self.build_tensor(angles_y_in_batch)[:, None, None]

        X,Y=self.XY_generation()
        phy = torch.atan2(Y, X)


        # Use broadcasting over cases and wavelengths.
        # k: (Num_wave*batch_size, 1, 1)
        # angles_x_rad: (Num_wave*batch_size, 1, 1, 1)
        # angles_y_rad: (Num_wave*batch_size, 1, 1, 1)
        k_b = k.repeat(batch_size,1,1)
        angles_x_rad_b = torch.repeat_interleave(angles_x_rad,Num_wave,dim=0)
        angles_y_rad_b = torch.repeat_interleave(angles_y_rad,Num_wave,dim=0)

        # X, Y: (Nx, Ny)
        # The incident phase now includes both x and y angle components.
        Inc_angle = torch.exp(1j * k_b * (torch.sin(angles_x_rad_b) * X + torch.sin(angles_y_rad_b) * Y))
        # Inc_angle represents the batched angular phase.

        # Generate the vortex phase.
        topos_tensor = self.build_tensor(topos_in_batch)[:, None, None]  # (batch_size, 1, 1)
        topos_tensor =torch.repeat_interleave(topos_tensor,Num_wave,dim=0)
        Inc_vortex = torch.exp(1j * topos_tensor * phy)  # (batch_size, Nx, Ny)

        # Combine angular and vortex phases.
        Inc = Inc_angle * Inc_vortex

        wavelength = self.flags.wavelength * batch_size

        return Inc, wavelength



    def make_loss(self, logit=None, labels=None, G=None, iter=0):
        if logit is None:
            return None

        mseloss = nn.MSELoss()

        eta=self.flags.eta

        # fujia=self.dirac_image()
        fujia=1
        MSE_loss = mseloss(fujia*logit, fujia*eta * labels)

        BDY_loss = 0

        # bb=torch.mean(torch.topk(torch.abs(self.field_on_second_layer[0]).flatten(), 30).values)
        # BDY_loss = relu(bb-15)/20
        # Image_NA_1=self.Gaussian_filter(torch.exp(1j * 2 * np.pi * self.MS_property[0][0]), self.flags.x, self.flags.y, self.flags.wavelength, 1.0)
        # Image_NA_2 = self.Gaussian_filter(torch.exp(1j * 2 * np.pi * self.MS_property[0][0]), self.flags.x,
        #                                   self.flags.y, self.flags.wavelength, 0.2)
        #
        # BDY_loss=torch.mean((torch.abs(Image_NA_1)**2-torch.abs(Image_NA_2)**2))/torch.mean(torch.abs(Image_NA_1)**2)*0.1

        if G is not None:  # This is using the boundary loss
            BDY_loss = G
        return torch.add(MSE_loss, BDY_loss)

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

        Num_topo = len(self.flags.topology_vortex)
        Num_wave = len(self.flags.wavelength)
        Num_angles = len(self.flags.Angle_x)  # Total number of angle pairs.

        total_cases = Num_wave * Num_topo * Num_angles


        image_t = None  # Accumulated target-image tensor.

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
            image = image0[np.newaxis, :, :]

            if i == 0:
                image_t = image
            else:
                image_t = np.concatenate([image_t, image], axis=0)

        result = self.build_tensor(image_t)
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

        # self.load()  # load the model as constructed
        # cuda = True if torch.cuda.is_available() else False
        # if cuda:
        #     self.model_NN.cuda()
        #
        # self.model_NN.eval()

        # Initialize the geometry_eval or the initial guess xs
        self.initialize_metasurface()
        # Set up the learning schedule and optimizer
        self.optm_eval = self.make_optimizer_eval()  # , optimizer_type='SGD')
        self.lr_scheduler = self.make_lr_scheduler(self.optm_eval)
        labels = self.read_figure()

        self.iteration = []
        self.save_loss = []

        # Split the label tensor by case, preserving the order in self.case_configs.
        num_output_channels_per_case = labels.shape[0] // len(self.case_configs)
        labels_per_case = torch.chunk(labels, len(self.case_configs), dim=0)

        for epoch in range(self.flags.train_step):
            self.optm_eval.zero_grad()
            total_loss_value = 0.0

            # Gradient-accumulation loop; each iteration processes one micro-batch.
            for i in range(self.flags.accumulation_steps):
                # Slice configurations and labels for the current micro-batch.
                start = i * self.flags.batch_size_per_gpu
                end = start + self.flags.batch_size_per_gpu

                micro_batch_configs = self.case_configs[start:end]
                if not micro_batch_configs: continue  # Skip an empty final micro-batch.

                micro_batch_labels_list = labels_per_case[start:end]
                micro_batch_labels = torch.cat(micro_batch_labels_list, dim=0)

                # Forward propagation.
                logit = self.model(
                    micro_batch_configs=micro_batch_configs,
                    apply_random_rotation=True,
                    apply_random_distance_error=True,
                    apply_random_angle_error=True,
                    rotation_error_mode=getattr(self.flags, 'rotation_error_mode', 'batch'),
                    distance_error_mode=getattr(self.flags, 'distance_error_mode', 'batch'),
                    angle_error_mode=getattr(self.flags, 'angle_error_mode', 'sample'),
                )
                logit = torch.abs(logit) ** 2

                # Compute the loss.
                # B_loss =self.loss_constraint()
                loss = self.make_loss(logit, micro_batch_labels, G=None, iter=epoch)/ self.flags.accumulation_steps

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
            num_micro_batches = (len(self.case_configs) + self.flags.batch_size_per_gpu - 1) // self.flags.batch_size_per_gpu

            for i in range(num_micro_batches):
                # Slice the current micro-batch configurations.
                start = i * self.flags.batch_size_per_gpu
                end = start + self.flags.batch_size_per_gpu

                micro_batch_configs = self.case_configs[start:end]
                if not micro_batch_configs:
                    continue  # Skip an empty final micro-batch.

                # Forward propagation.
                logit_batch = self.model(
                    micro_batch_configs=micro_batch_configs,
                    apply_random_rotation=False,
                    apply_random_distance_error=False,
                    apply_random_angle_error=False,
                    rotation_error_mode=getattr(self.flags, 'rotation_error_mode', 'batch'),
                    distance_error_mode=getattr(self.flags, 'distance_error_mode', 'batch'),
                    angle_error_mode=getattr(self.flags, 'angle_error_mode', 'sample'),
                )
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
            # phy = torch.rand([1, self.flags.Nx, self.flags.Ny])
            phy=self.build_tensor(phy,requires_grad=True)
            layer = [phy]
            output += layer
        self.MS_property = output

    def loss_constraint(self):
        relu = torch.nn.ReLU()

        size = len(self.MS_property)
        BDY_loss = 0
        for i in range(0, size):
            Wx_Wy_theta = self.MS_property[i]
            W_x = Wx_Wy_theta[0, :, :]
            W_y = Wx_Wy_theta[1, :, :]
            Theta = Wx_Wy_theta[2, :, :]
            BDY_loss_all = relu(W_x - 0.9) + relu(-W_x) + relu(W_y - 0.9) + relu(
                -W_y)  # ensure W_x W_y is between 0 and 1
            BDY_loss = BDY_loss + 5 * torch.mean(BDY_loss_all)

        return BDY_loss



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

        N, H, W = design_images.shape
        device = design_images.device
        dtype = measured_images.dtype
        epsilon = 1e-12

        if N == 0:
            return torch.empty((0, 0), device=device, dtype=dtype), 0.0

        # Build binary target masks T.
        # T_bool: (N, H, W)
        T_bool = design_images > threshold
        T_float = T_bool.to(dtype=dtype)

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
