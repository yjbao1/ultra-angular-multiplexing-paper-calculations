import torch
import numpy as np
from torch.fft import fft2, fftshift, ifft2, ifftshift


class Field_propagation():

    def __init__(self):
        self.flags = None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device=device

    def build_tensor(self, data, requires_grad=False):
        if isinstance(data, torch.Tensor):
            # clone() copies the data and detach() removes it from the existing graph so it is a leaf tensor.
            tensor = data.clone().detach().to(self.device, dtype=torch.float)
        else:
            # Convert NumPy arrays and Python lists to tensors.
            tensor = torch.tensor(data, device=self.device, dtype=torch.float)

        # Set gradient tracking after the tensor has been created.
        if requires_grad:
            tensor.requires_grad_(True)

        return tensor

        # return torch.tensor(nparray, requires_grad=requires_grad, device=self.device,dtype=torch.float)
        # Legacy RGB_split/python3 data did not apply an explicit dtype conversion.

    def func(self, src, wavelength=None, zp=None, refractive_index=None):
        if wavelength is None:
            wavelength = self.flags.wavelength
        if zp is None:
            zp = self.flags.distance

        if refractive_index is None:
            refractive_index = self.flags.refractive_index

        try:
            # Fall back to the aperture coordinates when observation coordinates are not provided.
            self.flags.x_ob
        except AttributeError:
            self.flags.x_ob = self.flags.x
            self.flags.y_ob = self.flags.y

        if self.flags.function == 'AS':
            return self.Angular_spectrum(src, x_ap=self.flags.x, y_ap=self.flags.y, x_ob=self.flags.x_ob,
                                         y_ob=self.flags.y_ob, zp=zp,
                                         wavelength=wavelength, refractive_index=refractive_index,
                                         NA=self.flags.NA)
        elif self.flags.function == 'Period':
            return self.Angular_spectrum_period(src, wavelength=wavelength, zp=zp,refractive_index=refractive_index)

        elif self.flags.function == 'gRS':
            return self.Convolution_rs(src, x_ap=self.flags.x, y_ap=self.flags.y, x_ob=self.flags.x_ob,
                                       y_ob=self.flags.y_ob,zp=zp,
                                       wavelength=wavelength, refractive_index=refractive_index)
        elif self.flags.function == 'Convolution_Fresnel':
            return self.Convolution_Fresnel(src, x_ap=self.flags.x, y_ap=self.flags.y, x_ob=self.flags.x_ob,
                                            y_ob=self.flags.y_ob, zp=zp,
                                         wavelength=wavelength, refractive_index=refractive_index)
        elif self.flags.function == 'Fresnel':
            return self.Fresnel(src, x_ap=self.flags.x, y_ap=self.flags.y, zp=zp,
                                         wavelength=wavelength, refractive_index=refractive_index)
        elif self.flags.function == 'FFT':
            return self.FFT(src, x_ap=self.flags.x, y_ap=self.flags.y, zp=zp,
                                         wavelength=wavelength, refractive_index=refractive_index)

        elif self.flags.function == 'IFFT':
            return self.IFFT(src, x_ap=self.flags.x, y_ap=self.flags.y, zp=zp, wavelength=wavelength,
                                refractive_index=refractive_index)

    def Angular_spectrum(self, src1, x_ap, y_ap, x_ob, y_ob, zp, wavelength, refractive_index, NA=None):  # NA is the numerical aperture in the propagation medium.
        Nx = src1.size(-2)
        Ny = src1.size(-1)
        NN = 2

        src = src1

        if src.ndim == 2:
            src = src[None, :, :]

        wavelength = self.build_tensor(wavelength)
        if wavelength.dim() > 0:
            wavelength = wavelength[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(wavelength.size(0), 1, 1)

        refractive_index = self.build_tensor(refractive_index)
        if refractive_index.dim() > 0:
            refractive_index = refractive_index[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(refractive_index.size(0), 1, 1)

        if NA is not None:
            NA = self.build_tensor(NA)
            if NA.dim() > 0:
                NA = NA[:, None, None]
                if src.size(0) == 1:
                    src = src.repeat(NA.size(0), 1, 1)

        zp = self.build_tensor(zp)
        if zp.dim() > 0:
            zp = zp[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(zp.size(0), 1, 1)

        try:
            fx_center = self.build_tensor(self.flags.fx_center)
            fy_center = self.build_tensor(self.flags.fy_center)
            if fx_center.dim() > 0:
                fx_center = fx_center[:, None, None]
                fy_center = fy_center[:, None, None]
                if src.size(0) == 1:
                    src = src.repeat(fx_center.size(0), 1, 1)
        except AttributeError:
            fx_center=0
            fy_center=0


        x_ap = self.build_tensor(x_ap)
        y_ap = self.build_tensor(y_ap)
        x_ob = self.build_tensor(x_ob)
        y_ob = self.build_tensor(y_ob)
        apsizex = x_ap[-1] - x_ap[0]
        apsizey = y_ap[-1] - y_ap[0]

        Tx = apsizex / Nx
        Ty = apsizey / Ny

        x_min=torch.min(x_ap[0],x_ob[0])
        x_max = torch.max(x_ap[-1], x_ob[-1])

        y_min = torch.min(y_ap[0], y_ob[0])
        y_max = torch.max(y_ap[-1], y_ob[-1])

        x_span = x_max - x_min
        y_span = y_max - y_min

        Nxx = torch.ceil(x_span / apsizex * Nx).int().item()
        Nyy = torch.ceil(y_span / apsizey * Ny).int().item()


        x_start = torch.round((x_ap[0] - x_min) / Tx).int().item()
        x_end = x_start + Nx

        y_start = torch.round((y_ap[0]  - y_min) / Ty).int().item()
        y_end = y_start + Ny

        if NA is None:
            deta_x = (NN - 1) * x_span
            NA = deta_x / torch.sqrt(deta_x ** 2 + zp ** 2)


        eval_batchsize = src.size(-3)
        U1 = torch.zeros(eval_batchsize,  Nxx,  Nyy, dtype=src.dtype, device=self.device)
        U1[:, x_start:x_end,y_start:y_end] = src
        U = torch.zeros(eval_batchsize, NN * Nxx, NN * Nyy, dtype=src.dtype, device=self.device)
        U[:,0:Nxx, 0:Nyy] = U1


        # sj = torch.linspace(-1 / Tx / 2, 1 / Tx / 2, NN * Nxx + 1, device=self.device)
        # nj = torch.linspace(-1 / Ty / 2, 1 / Ty / 2, NN * Nyy + 1, device=self.device)
        # sj = sj[0:-1]
        # nj = nj[0:-1]

        Px = NN * Nxx * Tx
        Py = NN * Nyy * Ty

        sj = -1 / 2 / Tx + torch.linspace(0, NN * Nxx - 1, NN * Nxx, device=self.device) / Px + 1 / 2 / Px * ((NN * Nxx )% 2)
        nj = -1 / 2 / Ty + torch.linspace(0, NN * Nyy - 1, NN * Nyy, device=self.device) / Py + 1 / 2 / Py * ((NN * Nyy )% 2)

        X, Y, = torch.meshgrid(sj, nj, indexing='ij')

        Uf = fftshift(fft2(ifftshift(U, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))


        # NA_range = (X ** 2 + Y ** 2) < NA ** 2 / (wavelength / refractive_index) ** 2
        NA_range = ((X - fx_center / wavelength) ** 2 + (Y - fy_center / wavelength) ** 2) < NA ** 2 / (wavelength / refractive_index) ** 2

        trans_v = (wavelength / refractive_index) ** (-2) - X ** 2 - Y ** 2
        trans_v = trans_v.type(torch.complex64)
        trans = torch.exp(1j * 2 * np.pi * zp * torch.sqrt(trans_v))
        result = Uf * trans *NA_range
        II = fftshift(ifft2(ifftshift(result, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))

        x_start_ob = torch.round((x_ob[0] - x_min) / Tx).int()
        x_end_ob = torch.round((x_ob[-1] - x_min) / Tx).int()

        y_start_ob = torch.round((y_ob[0] - y_min) / Ty).int()
        y_end_ob = torch.round((y_ob[-1] - y_min) / Ty).int()

        I = II[:, 0: Nxx, 0:Nyy]
        I=I[:,x_start_ob:x_end_ob,y_start_ob:y_end_ob]

        if  src1.ndim ==2:
            I=torch.squeeze(I)

        return I

    def Angular_spectrum_period(self, src1,wavelength=None, zp=None,refractive_index=None,NA=0.95):
        Nx = src1.size(-2)
        Ny = src1.size(-1)

        src = src1

        if src.ndim == 2:
            src = src[None, :, :]

        if wavelength is None:
            wavelength = self.flags.wavelength

        wavelength = self.build_tensor(wavelength)
        if wavelength.dim() > 0:
            wavelength = wavelength[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(wavelength.size(0), 1, 1)

        if refractive_index is None:
            refractive_index = self.flags.refractive_index

        if zp is None:
            zp = self.flags.distance


        x_ap = self.build_tensor(self.flags.x)
        y_ap = self.build_tensor(self.flags.y)
        Px = x_ap[-1] - x_ap[0]
        Py = y_ap[-1] - y_ap[0]

        Tx = Px / Nx
        Ty = Py / Ny


        sj = -1 / 2 / Tx + torch.linspace(0,  Nx - 1,  Nx, device=self.device) / Px + 1 / 2 / Px * (( Nx) % 2)
        nj = -1 / 2 / Ty + torch.linspace(0,  Ny - 1, Ny, device=self.device) / Py + 1 / 2 / Py * (( Ny) % 2)

        X, Y, = torch.meshgrid(sj, nj, indexing='ij')

        Uf = fftshift(fft2(ifftshift(src, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))

        NA_range = (X ** 2 + Y ** 2) < NA ** 2 / (wavelength / refractive_index) ** 2

        trans_v = (wavelength / refractive_index) ** (-2) - X ** 2 - Y ** 2
        trans_v = trans_v.type(torch.complex64)
        trans = torch.exp(1j * 2 * np.pi * zp * torch.sqrt(trans_v))
        result1 = Uf * trans
        result = result1 * NA_range
        I = fftshift(ifft2(ifftshift(result, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))

        if src1.ndim == 2:
            I = torch.squeeze(I)

        return I

    def Gaussian_filter(self, src, x_ap, y_ap, wavelength, NA):
        return self.Angular_spectrum(src, x_ap, y_ap, x_ob=x_ap, y_ob=y_ap, zp=0, wavelength=wavelength, refractive_index=1,
                                     NA=NA)

    def Convolution_rs(self, src1, x_ap, y_ap, x_ob, y_ob, zp, wavelength, refractive_index):

        src=src1

        if src.ndim == 2:
           src=src[None,:,:]

        wavelength = self.build_tensor(wavelength)
        if wavelength.dim() > 0:
            wavelength = wavelength[:, None, None]
            if src.size(0)==1:
                src=src.repeat(wavelength.size(0),1,1)

        zp = self.build_tensor(zp)
        if zp.dim() > 0:
            zp = zp[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(zp.size(0), 1, 1)

        refractive_index = self.build_tensor(refractive_index)
        if refractive_index.dim() > 0:
            refractive_index = refractive_index[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(refractive_index.size(0), 1, 1)


        k = 2 * np.pi / wavelength * refractive_index
        x_ap = self.build_tensor(x_ap)
        y_ap = self.build_tensor(y_ap)
        x_ob = self.build_tensor(x_ob)
        y_ob = self.build_tensor(y_ob)

        apsizex = x_ap[-1] - x_ap[0]
        apsizey = y_ap[-1] - y_ap[0]
        obsizex = x_ob[-1] - x_ob[0]
        obsizey = y_ob[-1] - y_ob[0]

        ratio_x = obsizex / apsizex
        ratio_x_int = torch.ceil(ratio_x)

        ratio_y = obsizey / apsizey
        ratio_y_int = torch.ceil(ratio_y)

        Nx1 = src.size(-2)  # for torch tensor only
        Ny1 = src.size(-1)

        Num_x = int(ratio_x * Nx1)
        Num_y = int(ratio_y * Ny1)

        # sj = torch.linspace(min(x_ap), max(x_ap), Nx1 + 1, device=self.device)
        sj = torch.linspace(x_ap[0], x_ap[-1], Nx1 + 1, device=self.device)  # this is much faster
        sj = sj[:-1]
        dsj = sj[1] - sj[0]
        nj = torch.linspace(y_ap[0], y_ap[-1], Ny1 + 1, device=self.device)
        nj = nj[:-1]
        dnj = nj[1] - nj[0]

        eval_batchsize = src.size(-3)
        I = torch.zeros(eval_batchsize, int(ratio_x_int * Nx1), int(ratio_y_int * Ny1), dtype=torch.complex64,
                        device=self.device)
        U = torch.zeros(eval_batchsize, 2 * Nx1 - 1, 2 * Ny1 - 1, dtype=torch.complex64, device=self.device)
        U[:, 0:Nx1, 0:Ny1] = src


        U_ = fft2(U, dim=(-1, -2))

        ratio_x_int = ratio_x_int.to(torch.int8)
        ratio_y_int = ratio_y_int.to(torch.int8)

        for nx in range(0, ratio_x_int):
            for ny in range(0, ratio_y_int):
                # xj = torch.linspace(min(x_ob), max(x_ob), Nx1 + 1, device=self.device)
                xj = torch.linspace(x_ob[0] + apsizex * nx, x_ob[0] + apsizex * (nx + 1), Nx1 + 1,
                                    device=self.device)  # this is much faster
                xj = xj[:-1]
                # # yj = torch.linspace(min(y_ob), max(y_ob), Ny1 + 1, device=self.device)
                yj = torch.linspace(y_ob[0] + apsizey * ny, y_ob[0] + apsizey * (ny + 1), Ny1 + 1,
                                    device=self.device)  # this is much faster
                yj = yj[:-1]

                # Xj = torch.linspace(min(xj) - max(sj), max(xj) - min(sj), 2 * Nx1 - 1, device=self.device)
                # Yj = torch.linspace(min(yj) - max(nj), max(yj) - min(nj), 2 * Ny1 - 1, device=self.device)

                Xj = torch.linspace(xj[0] - sj[-1], xj[-1] - sj[0], 2 * Nx1 - 1, device=self.device)
                Yj = torch.linspace(yj[0] - nj[-1], yj[-1] - nj[0], 2 * Ny1 - 1, device=self.device)

                H = Field_propagation.gRS(Xj, Yj, zp, k)

                S = ifft2(U_ * fft2(H)) * dsj * dnj


                I[:, nx * Nx1:(nx + 1) * Nx1, ny * Ny1:(ny + 1) * Ny1] = S[:, Nx1 - 1:2 * Nx1 - 1,
                                                                             Ny1 - 1:2 * Ny1 - 1]


        I = I[:, 0:Num_x, 0:Num_y]

        if  src1.ndim ==2:
            I=torch.squeeze(I)

        return I


    def Convolution_Fresnel(self, src1, x_ap, y_ap, x_ob, y_ob, zp, wavelength, refractive_index):

        src = src1

        if src.ndim == 2:
            src = src[None, :, :]

        wavelength = self.build_tensor(wavelength)
        if wavelength.dim() > 0:
            wavelength = wavelength[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(wavelength.size(0), 1, 1)

        zp = self.build_tensor(zp)
        if zp.dim() > 0:
            zp = zp[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(zp.size(0), 1, 1)

        k = 2 * np.pi / wavelength * refractive_index
        x_ap = self.build_tensor(x_ap)
        y_ap = self.build_tensor(y_ap)
        x_ob = self.build_tensor(x_ob)
        y_ob = self.build_tensor(y_ob)

        Nx1 = src1.size(-2)  # for torch tensor only
        Ny1 = src1.size(-1)

        tt = zp * wavelength / refractive_index

        ssx = (x_ob[-1] - x_ob[0]) / (x_ap[-1] - x_ap[0])
        ssy = (y_ob[-1] - y_ob[0]) / (y_ap[-1] - y_ap[0])

        xa = torch.linspace(x_ap[0], x_ap[-1], Nx1 + 1, device=self.device)  # this is much faster
        xa = xa[:-1]
        dx = xa[1] - xa[0]
        ya = torch.linspace(y_ap[0], y_ap[-1], Ny1 + 1, device=self.device)
        ya = ya[:-1]
        dy = ya[1] - ya[0]

        xo = torch.linspace(x_ob[0] / ssx, x_ob[-1] / ssx, Nx1 + 1, device=self.device)  # this is much faster
        xo = xo[:-1]
        x = ssx * xo
        yo = torch.linspace(y_ob[0] / ssy, y_ob[-1] / ssy, Ny1 + 1, device=self.device)
        yo = yo[:-1]
        y = ssy * yo

        x1 = torch.linspace(xo[0] - xa[-1], xo[-1] - xa[0], 2 * Nx1 - 1, device=self.device)  # this is much faster
        y1 = torch.linspace(yo[0] - ya[-1], yo[-1] - ya[0], 2 * Ny1 - 1, device=self.device)  # this is much faster

        Xa, Ya = torch.meshgrid(xa, ya, indexing='ij')

        Xo, Yo = torch.meshgrid(xo, yo, indexing='ij')

        X1, Y1 = torch.meshgrid(x1, y1, indexing='ij')

        X, Y = torch.meshgrid(x, y, indexing='ij')

        ratio1 = torch.exp(1j * k / 2 / zp * Xa ** 2) * torch.exp(1j * k / 2 / zp * Ya ** 2)
        ratio2 = torch.exp(-1j * np.pi * ssx / tt * Xa ** 2) * torch.exp(-1j * np.pi * ssy / tt * Ya ** 2)
        ratio3 = torch.exp(-1j * np.pi * ssx / tt * Xo ** 2) * torch.exp(-1j * np.pi * ssy / tt * Yo ** 2)
        ratio4 = torch.exp(1j * k * zp) / (1j * tt) * torch.exp(1j * k / 2 / zp * (X ** 2 + Y ** 2))

        gx = torch.exp(1j * np.pi * ssx / tt * X1 ** 2) * torch.exp(1j * np.pi * ssy / tt * Y1 ** 2)

        eval_batchsize = src.size(-3)

        U1 = torch.zeros(eval_batchsize, 2 * Nx1 - 1, 2 * Ny1 - 1, dtype=torch.complex64, device=self.device)
        U1[:, 0:Nx1, 0:Ny1] = src * ratio1 * ratio2


        U_ = fft2(U1, dim=(-1, -2))
        Gx = fft2(gx, dim=(-1, -2))

        U2 = ifft2(U_ * Gx, dim=(-1, -2)) * dx * dy

        I1 = U2[:, Nx1 - 1:2 * Nx1 - 1, Ny1 - 1:2 * Ny1 - 1]
        I = I1 * ratio3 * ratio4

        if  src1.ndim ==2:
            I=torch.squeeze(I)

        return I

    def Fresnel(self, src1, x_ap, y_ap, zp, wavelength, refractive_index):

        src = src1

        if src.ndim == 2:
            src = src[None, :, :]

        wavelength = self.build_tensor(wavelength)
        if wavelength.dim() > 0:
            wavelength = wavelength[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(wavelength.size(0), 1, 1)

        zp = self.build_tensor(zp)
        if zp.dim() > 0:
            zp = zp[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(zp.size(0), 1, 1)


        wavelength1 = wavelength / refractive_index

        k = 2 * np.pi / wavelength1
        x_ap = self.build_tensor(x_ap)
        y_ap = self.build_tensor(y_ap)

        Nx1 = src.size(-2)  # for torch tensor only
        Ny1 = src.size(-1)

        Lx_0 = x_ap[-1] - x_ap[0]
        Ly_0 = y_ap[-1] - y_ap[0]


        sjj = torch.linspace(x_ap[0], x_ap[-1], Nx1 + 1, device=self.device)  # this is much faster
        sjj = sjj[:-1]

        njj = torch.linspace(y_ap[0], y_ap[-1], Ny1 + 1, device=self.device)
        njj = njj[:-1]

        xx, yy = torch.meshgrid(sjj, njj, indexing='ij')

        Fresnel_kernel  = torch.exp(1j * k / 2 / zp * (xx ** 2 + yy ** 2))



        Lx = wavelength1 * zp * Nx1 / Lx_0
        Ly = wavelength1 * zp * Ny1 / Ly_0

        xj_base = torch.linspace(-0.5, 0.5, Nx1+1, device=self.device)
        xj_base = xj_base[:-1]
        yj_base = torch.linspace(-0.5, 0.5, Ny1+1, device=self.device)
        yj_base = yj_base[:-1]

        x_t_base, y_t_base = torch.meshgrid(xj_base, yj_base, indexing='ij')

        x_t = x_t_base.unsqueeze(0) * Lx
        y_t = y_t_base.unsqueeze(0) * Ly

        phase = torch.exp(1j * k * zp) / (1j * wavelength1 * zp) * torch.exp(1j * k / 2 / zp * (x_t ** 2 + y_t ** 2))


        f2 = src1 * Fresnel_kernel

        Uf = fftshift(fft2(fftshift(f2, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))

        Uf = Uf * phase
        Tx = Lx_0 / Nx1
        Ty = Ly_0 / Ny1
        I = Uf * Tx * Ty

        Lx = Lx.cpu().data.numpy()
        Ly = Ly.cpu().data.numpy()
        # xp = np.array([-Lx / 2, Lx / 2])
        # yp = np.array([-Ly / 2, Ly / 2])

        # return I, xp, yp
        if  src1.ndim ==2:
            I=torch.squeeze(I)


        return I

    def FFT(self, src1, x_ap, y_ap, zp, wavelength, refractive_index):
        src = src1

        if src.ndim == 2:
            src = src[None, :, :]

        wavelength = self.build_tensor(wavelength)
        if wavelength.dim() > 0:
            wavelength = wavelength[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(wavelength.size(0), 1, 1)

        zp = self.build_tensor(zp)
        if zp.dim() > 0:
            zp = zp[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(zp.size(0), 1, 1)

        wavelength1 = wavelength / refractive_index
        k = 2 * np.pi / wavelength1
        x_ap = self.build_tensor(x_ap)
        y_ap = self.build_tensor(y_ap)

        Nx1 = src.size(-2)  # for torch tensor only
        Ny1 = src.size(-1)

        Lx_0 = x_ap[-1] - x_ap[0]
        Ly_0 = y_ap[-1] - y_ap[0]

        Lx = wavelength1 * zp * Nx1 / Lx_0
        Ly = wavelength1 * zp * Ny1 / Ly_0

        Uf = fftshift(fft2(fftshift(src, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))

        Tx = Lx_0 / Nx1
        Ty = Ly_0 / Ny1
        I = Uf * Tx * Ty
        I = I * torch.exp(1j * 2 * k * zp) / (1j * wavelength1 * zp)

        Lx=Lx[:,0,0]
        Ly=Ly[:,0,0]

        Lx = Lx.cpu().data.numpy()
        Ly = Ly.cpu().data.numpy()


        t_x = np.linspace(-0.5, 0.5, Nx1 + 1)[:-1]  # Shape: (Nx1,)
        t_y = np.linspace(-0.5, 0.5, Ny1 + 1)[:-1]  # Shape: (Ny1,)

        # Use NumPy broadcasting for the coordinate scaling.
        # Lx[:, None] changes the shape from (B,) to (B, 1).
        # Multiplying (B, 1) by (Nx1,) broadcasts to a (B, Nx1) matrix.
        xp = Lx[:, None] * t_x
        yp = Ly[:, None] * t_y

        if xp.shape[0] == 1:
            xp = np.squeeze(xp, axis=0)
            yp = np.squeeze(yp, axis=0)


        self.xp=xp
        self.yp =yp

        if  src1.ndim ==2:
            I=torch.squeeze(I)

        return I


    def IFFT(self, src1, x_ap, y_ap, zp, wavelength, refractive_index):

        src = src1

        if src.ndim == 2:
            src = src[None, :, :]

        wavelength = self.build_tensor(wavelength)
        if wavelength.dim() > 0:
            wavelength = wavelength[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(wavelength.size(0), 1, 1)

        zp = self.build_tensor(zp)
        if zp.dim() > 0:
            zp = zp[:, None, None]
            if src.size(0) == 1:
                src = src.repeat(zp.size(0), 1, 1)

        wavelength1 = wavelength / refractive_index
        k = 2 * np.pi / wavelength1
        x_ap = self.build_tensor(x_ap)
        y_ap = self.build_tensor(y_ap)

        Nx1 = src.size(-2)  # for torch tensor only
        Ny1 = src.size(-1)

        Lx_0 = x_ap[-1] - x_ap[0]
        Ly_0 = y_ap[-1] - y_ap[0]

        Lx = wavelength1 * zp * Nx1 / Lx_0
        Ly = wavelength1 * zp * Ny1 / Ly_0

        phase = torch.exp(-1j * 2* k * zp) * (1j * wavelength1 * zp)

        Uf = ifftshift(ifft2(ifftshift(src, dim=(-1, -2)), dim=(-1, -2)), dim=(-1, -2))

        Uf = Uf * phase
        Tx = Lx / Nx1
        Ty = Ly / Ny1
        I = Uf / Tx / Ty

        Lx=Lx[:,0,0]
        Ly=Ly[:,0,0]
        Lx = Lx.cpu().data.numpy()
        Ly = Ly.cpu().data.numpy()
        xp = np.linspace(-Lx / 2, Lx / 2,Nx1+1)
        self.xp=xp[:-1]
        yp = np.linspace(-Ly / 2, Ly / 2,Ny1+1)
        self.yp = yp[:-1]


        if  src1.ndim ==2:
            I=torch.squeeze(I)


        return I



    @staticmethod
    def gRS(x, y, z, k):
        X, Y, = torch.meshgrid(x, y, indexing='ij')
        # X = X.T  # PyTorch does not require the transpose here.
        # Y = Y.T
        r = torch.sqrt(X ** 2 + Y ** 2 + z ** 2)


        return torch.exp(1j * k * r) * (z / r ** 3) / (2 * np.pi) * (1 - 1j * k * r)  # k=2*pi/lambda*n



    @staticmethod
    def numpy_to_list(array):
        list_numpy = []
        if np.ndim(array) == 3:
            Nx = np.size(array, 0)
            for i in range(0, Nx):
                list_numpy.append(array[i])
        if np.ndim(array) == 2:
            list_numpy.append(array)
        return list_numpy

    @staticmethod
    def list_from_gpu_to_cpu(list_gpu):
        list_cpu = []
        for i, list1 in enumerate(list_gpu):
            if list1.dtype == torch.complex64:
                list1 = list1.to(torch.complex128)  # the fdtd can not load complex64， have to convert to complex128
            if list1.dtype == torch.float32:
                list1 = list1.to(torch.float64)  # the fdtd can not load float32， have to convert to float64
            list_cpu.append(list1.cpu().data.numpy())

        return list_cpu
