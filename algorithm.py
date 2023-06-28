import torch, math
from torch import nn


class DPAC(nn.Module):
    """Double-phase Amplitude Coding

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param device: torch.device

    Usage
    -----
    Functions as a pytorch module:

    >>> dpac = DPAC(...)
    >>> _, final_phase = dpac(target_amp, target_phase)

    target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    target_amp (optional): phase at the target plane, with dimensions [batch, 1, height, width]
    final_phase: optimized phase-only representation at SLM plane, same dimensions

    """
    def __init__(self, prop_dist, wavelength, feature_size, prop_model='ASM', propagator=None,
                 device=torch.device('cuda')):
        """

        """
        super(DPAC, self).__init__()

        # propagation is from target to SLM plane (one step)
        self.prop_dist = -prop_dist
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.precomputed_H = None
        self.prop_model = prop_model
        self.prop = propagator
        self.dev = device

    def forward(self, target_amp, target_phase=None):
        if target_phase is None:
            target_phase = torch.zeros_like(target_amp)

        if self.precomputed_H is None and self.prop_model == 'ASM':
            self.precomputed_H = self.prop(torch.empty(*target_amp.shape, dtype=torch.complex64), self.feature_size,
                                           self.wavelength, self.prop_dist, return_H=True)
            self.precomputed_H = self.precomputed_H.to(self.dev).detach()
            self.precomputed_H.requires_grad = False

        final_phase = double_phase_amplitude_coding(target_phase, target_amp, self.prop_dist,
                                                    self.wavelength, self.feature_size,
                                                    prop_model=self.prop_model, propagator=self.prop,
                                                    precomputed_H=self.precomputed_H)
        return None, final_phase

    @property
    def phase_path(self):
        return self._phase_path

    @phase_path.setter
    def phase_path(self, phase_path):
        self._phase_path = phase_path

def double_phase_amplitude_coding(target_phase, target_amp, prop_dist, wavelength, feature_size,
                                  prop_model='ASM', propagator=None,
                                  dtype=torch.float32, precomputed_H=None):
    """
    Use a single propagation and converts amplitude and phase to double phase coding

    Input
    -----
    :param target_phase: The phase at the target image plane
    :param target_amp: A tensor, (B,C,H,W), the amplitude at the target image plane.
    :param prop_dist: propagation distance, in m.
    :param wavelength: wavelength, in m.
    :param feature_size: The SLM pixel pitch, in meters.
    :param prop_model: The light propagation model to use for prop from target plane to slm plane
    :param propagator: propagation_ASM
    :param dtype: torch datatype for computation at different precision.
    :param precomputed_H: pre-computed kernel - to make it faster over multiple iteration/images - calculate it once

    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the shape of (1,1,H,W)
    """
    real, imag = polar_to_rect(target_amp, target_phase)
    target_field = torch.complex(real, imag)

    slm_field = propagate_field(target_field, propagator, prop_dist, wavelength, feature_size,
                                      prop_model, dtype, precomputed_H)

    slm_phase = double_phase(slm_field, three_pi=False, mean_adjust=True)

    return slm_phase


def double_phase(field, three_pi=True, mean_adjust=True):
    """Converts a complex field to double phase coding

    field: A complex64 tensor with dims [..., height, width]
    three_pi, mean_adjust: see double_phase_amp_phase
    """
    return double_phase_amp_phase(field.abs(), field.angle(), three_pi, mean_adjust)

def double_phase_amp_phase(amplitudes, phases, three_pi=True, mean_adjust=True):
    """converts amplitude and phase to double phase coding

    amplitudes:  per-pixel amplitudes of the complex field
    phases:  per-pixel phases of the complex field
    three_pi:  if True, outputs values in a 3pi range, instead of 2pi
    mean_adjust:  if True, centers the phases in the range of interest
    """
    # normalize
    amplitudes = amplitudes / amplitudes.max()
    amplitudes = torch.clamp(amplitudes, -0.99999, 0.99999)

    # phases_a = phases - torch.acos(amplitudes)
    # phases_b = phases + torch.acos(amplitudes)
    
    acos_amp = torch.acos(amplitudes)
    phases_a = phases - acos_amp
    phases_b = phases + acos_amp

    phases_out = phases_a
    phases_out[..., ::2, 1::2] = phases_b[..., ::2, 1::2]
    phases_out[..., 1::2, ::2] = phases_b[..., 1::2, ::2]

    if three_pi:
        max_phase = 3 * math.pi
    else:
        max_phase = 2 * math.pi

    if mean_adjust:
        phases_out -= phases_out.mean()

    return (phases_out + max_phase / 2) % max_phase - max_phase / 2



# utils
def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag


def replace_amplitude(field, amplitude):
    """takes a Complex tensor with real/imag channels, converts to
    amplitude/phase, replaces amplitude, then converts back to real/imag

    resolution of both Complex64 tensors should be (M, N, height, width)
    """
    # replace amplitude with target amplitude and convert back to real/imag
    real, imag = polar_to_rect(amplitude, field.angle())

    # concatenate
    return torch.complex(real, imag)


def propagate_field(input_field, propagator, prop_dist=0.2, wavelength=520e-9, feature_size=(6.4e-6, 6.4e-6),
                    prop_model='ASM', dtype=torch.float32, precomputed_H=None):
    """
    A wrapper for various propagation methods, including the parameterized model.
    Note that input_field is supposed to be in Cartesian coordinate, not polar!

    Input
    -----
    :param input_field: pytorch complex tensor shape of (1, C, H, W), the field before propagation, in X, Y coordinates
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength of the wave in m.
    :param feature_size: pixel pitch
    :param prop_model: propagation model ('ASM', 'MODEL', 'fresnel', ...)
    :param trained_model: function or model instance for propagation
    :param dtype: torch.float32 by default
    :param precomputed_H: Propagation Kernel in Fourier domain (could be calculated at the very first time and reuse)

    Output
    -----
    :return: output_field: pytorch complex tensor shape of (1, C, H, W), the field after propagation, in X, Y coordinates
    """

    if prop_model == 'ASM':
        output_field = propagator(u_in=input_field, z=prop_dist, feature_size=feature_size, wavelength=wavelength,
                                  dtype=dtype, precomped_H=precomputed_H)
    else:
        raise ValueError('Unexpected prop_model value')

    return output_field