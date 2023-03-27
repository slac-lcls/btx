import numpy as np
from matplotlib import pyplot as plt

class epix10k2M_calibration:
    """
    Class to explore epix10k2M calibration behaviour.
    """
    def __init__(self):
        print('epix10k2M_calibration')
        self.limit_14bit = 16383
        self.gain_list = ['high', 'medium', 'low']
        self.set_gain_color()
        self.set_gain_adu_per_keV()
        self.set_pedestal_adu()
        self.set_offset_adu()
        self.set_noise_keV()
        self.set_saturation_keV()
        self.set_switch_fraction()

    def set_gain_color(self, gain=None, color='gray'):
        """
        Assign colors to gain levels. For plotting purposes.

        Parameters
        ----------
        gain : str
            if None, all 'high', 'medium' and 'low' gain colors are set.
        color : str
            color to use for gain if gain is not None.
        """
        if gain is None:
            self.gain_color = {}
            self.gain_color['high'] = 'green'
            self.gain_color['medium'] = 'blue'
            self.gain_color['low'] = 'red'
        else:
            self.gain_color[gain] = color
        print(f'Gain color: {self.gain_color}')

    def set_gain_adu_per_keV(self, gain=None, value=0.):
        """
        Assign the ADU per 9.5 keV gains values for the epix10k2M.
        Default values taken from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7206547/

        Parameters
        ----------
        gain : str
            if None, all 'high', 'medium' and 'low' gain colors are set.
        value : float
            value to set gain if gain is not None.
        """
        if gain is None:
            self.gain_adu_per_keV = {}
            self.gain_adu_per_keV['high'] = 162 / 9.5
            self.gain_adu_per_keV['medium'] = 48.6 / 9.5
            self.gain_adu_per_keV['low'] = 1.62 / 9.5
        else:
            self.gain_adu_per_keV[gain] = value
        print(f'Gain ADU/keV: {self.gain_adu_per_keV}')

    def set_pedestal_adu(self, gain=None, value=0.):
        """
        Assign gain pedestals for the epix10k2M.
        Default value is 3100 ADU. See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7206547/

        Parameters
        ----------
        gain : str
            if None, all 'high', 'medium' and 'low' gain colors are set.
        value : float
            value to set pedestal for gain if gain is not None.
        """
        if gain is None:
            self.pedestal_adu = {}
            self.pedestal_adu['high'] = 3100
            self.pedestal_adu['medium'] = 3100
            self.pedestal_adu['low'] = 3100
        else:
            self.pedestal_adu[gain] = value
        print(f'Pedestal in ADU: {self.pedestal_adu}')

    def set_offset_adu(self, gain=None, value=0.):
        """
        Assign gain offsets for the epix10k2M.
        Default value is 0 ADU. See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7206547/

        Parameters
        ----------
        gain : str
            if None, all 'high', 'medium' and 'low' gain colors are set.
        value : float
            value to set offset for gain if gain is not None.
        """
        if gain is None:
            self.offset_adu = {}
            self.offset_adu['high'] = 0
            self.offset_adu['medium'] = 0
            self.offset_adu['low'] = 0
        else:
            self.offset_adu[gain] = value
        print(f'Offset in ADU: {self.offset_adu}')

    def set_noise_keV(self, gain=None, value=0.):
        """
        Assign gain noise for the epix10k2M.
        Default value are taken from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7206547/

        Parameters
        ----------
        gain : str
            if None, all 'high', 'medium' and 'low' gain colors are set.
        value : float
            value to set noise for gain if gain is not None.
        """
        if gain is None:
            self.noise_keV = {}
            self.noise_keV['high'] = 0.325
            self.noise_keV['medium'] = 0.562
            self.noise_keV['low'] = 13
        else:
            self.noise_keV[gain] = value
        print(f'Noise in keV: {self.noise_keV}')

    def set_saturation_keV(self, gain=None, value=0.):
        """
        Assign gain saturation levels for the epix10k2M.
        Default values are taken from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7206547/

        Parameters
        ----------
        gain : str
            if None, all 'high', 'medium' and 'low' gain colors are set.
        value : float
            value to set saturation for gain if gain is not None.
        """
        if gain is None:
            self.saturation_keV = {}
            self.saturation_keV['high'] = 80 * 9.5
            self.saturation_keV['medium'] = 270 * 9.5
            self.saturation_keV['low'] = 8200 * 9.5
        else:
            self.saturation_keV[gain] = value
        print(f'Saturation in keV: {self.saturation_keV}')

    def set_switch_fraction(self, value=0.8):
        """
        Defines fraction of saturation level at which to switch gain.

        Parameters
        ----------
        value : float between 0.0 and 1.0
        """
        self.switch_fraction = value

    def adu_to_keV(self, adu, gain, offset=False):
        """
        Converts ADU value to keV for a given gain level, accounting for pedestal.
        Optionally accounts for offset.
        """
        keV = (adu - self.pedestal_adu[gain]) / self.gain_adu_per_keV[gain]
        if offset:
            keV -= self.offset_adu[gain] / self.gain_adu_per_keV[gain]
        keV[np.where(keV > self.saturation_keV[gain])] = self.saturation_keV[gain]
        return keV

    def switch_adu(self, keV, gain):
        """
        Switch to next gain level if photon energy in keV reaches a given fraction of the saturation energy.
        Returns the ADU value at which to switch gain.
        """
        switch = np.argwhere(keV[gain] > self.switch_fraction * self.saturation_keV[gain])
        if len(switch) == 0:
            switch = [[self.limit_14bit]]
        return switch[0][0]

    def plot_calibration(self, gains=None):
        raw_adu = np.arange(0, self.limit_14bit, 1)
        calibrated_no_offset_keV = {}
        calibrated_keV = {}
        for gain in self.gain_list:
            calibrated_no_offset_keV[gain] = self.adu_to_keV(raw_adu, gain)
            calibrated_keV[gain] = self.adu_to_keV(raw_adu, gain, offset=True)
        #
        if gains is None:
            gains = self.gain_list
        fig = plt.figure(figsize=(4, 5), dpi=200)
        for gain in gains:
            plt.plot(raw_adu, calibrated_no_offset_keV[gain], label=f'{gain} without offset',
                     color=self.gain_color[gain], linewidth=0.5, linestyle='dashed')
            if self.offset_adu[gain] != 0:
                plt.plot(raw_adu, calibrated_keV[gain], label=f'{gain} with {self.offset_adu[gain]} ADU offset',
                         color=self.gain_color[gain], linewidth=0.5)
            plt.axhspan(self.noise_keV[gain], self.saturation_keV[gain], alpha=0.1, color=self.gain_color[gain],
                        label=f'{gain} noise-to-saturation')
            plt.axhline(self.switch_fraction * self.saturation_keV[gain], color=self.gain_color[gain],
                        label=f'{gain} switch', linewidth=0.5, linestyle='dotted')
            plt.axvspan(0, self.pedestal_adu[gain], alpha=0.5, color=self.gain_color[gain],
                        label=f'pedestal ({gain}) = {self.pedestal_adu[gain]} ADU')
        plt.axhline(0, color='black')
        plt.yscale('symlog')
        plt.xlim(0, self.limit_14bit)
        plt.ylim(-1e4, 1e5)
        plt.xlabel('Raw (ADU)')
        plt.ylabel('Calibrated (keV)')
        # plt.grid()
        plt.legend(bbox_to_anchor=(1, 1))
        plt.show()

    def plot_simulation(self, vmin=None, vmax=None, vstep=10, log=False):
        if vmin is None:
            vmin = self.pedestal_adu['medium']
        if vmax is None:
            vmax = 0.75 * self.limit_14bit
        # sampled_adu = self.pedestal_adu['medium'] + np.arange(0, 0.75*self.limit_14bit, 10) # X-ray generated ADU, on top of background voltage
        sampled_adu = vmin + np.arange(0, vmax, vstep)  # X-ray generated ADU, on top of background voltage
        print(f'Sampling incoming intensity from {int(np.min(sampled_adu))} to {int(np.max(sampled_adu))} ADU')

        sampled_gain = np.zeros_like(sampled_adu)
        switched_adu = np.copy(sampled_adu)
        switched_keV = np.zeros_like(sampled_adu)

        # find switching and saturation ADU values from a first keV pass
        calibrated_keV = {}
        for gain in self.gain_list:
            calibrated_keV[gain] = self.adu_to_keV(sampled_adu, gain)
        switch_medium_to_low_adu = sampled_adu[self.switch_adu(calibrated_keV, 'medium')]
        saturated_low_adu = sampled_adu[self.switch_adu(calibrated_keV, 'low')]
        print(f'Switching from Medium to Low at {int(switch_medium_to_low_adu)} ADU')
        print(f'Saturating Low at {int(saturated_low_adu)} ADU')

        sampled_gain[np.where(sampled_adu > switch_medium_to_low_adu)] = 1  # switch to low

        switched_adu[np.where(sampled_gain == 1)] += self.pedestal_adu['low'] - switch_medium_to_low_adu

        switched_keV[np.where(sampled_gain == 0)] = self.adu_to_keV(switched_adu[np.where(sampled_gain == 0)], 'medium',
                                                                    True)
        switched_keV[np.where(sampled_gain == 1)] = self.adu_to_keV(switched_adu[np.where(sampled_gain == 1)], 'low',
                                                                    True)

        plt.title('Simulated calibration')
        plt.scatter(switched_adu, switched_keV, marker='.', c=sampled_adu)
        plt.xlabel('Raw (ADU)')
        plt.ylabel('Calibrated (keV)')
        plt.xlim(0, self.limit_14bit)
        plt.grid()
        if log:
            plt.ylim(-1e4, 1e6)
            plt.yscale('symlog')
        plt.colorbar(label='Incoming (ADU)')
        plt.show()