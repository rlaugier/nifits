import warnings
from unittest import TestCase

import astropy.units as u
import numpy as np
import sympy as sp
from erfa import ErfaWarning

import nifits.io.oifits as io


class BaseTestCase(TestCase):
    """
    Base class for all test cases.
    Provides common setup and teardown methods.
    """

    def setUp(self):
        """
        Set up the test environment.
        This method is called before each test case.
        """
        # Set up an example NIFITS object

        # Set up combiner
        mat_3T_txt = """
        Matrix([
        [sqrt(3)/3,                sqrt(3)/3,                sqrt(3)/3],
        [sqrt(3)/3,  sqrt(3)*exp(2*I*pi/3)/3, sqrt(3)*exp(-2*I*pi/3)/3],
        [sqrt(3)/3, sqrt(3)*exp(-2*I*pi/3)/3,  sqrt(3)*exp(2*I*pi/3)/3]])
        """
        combiner_s = sp.sympify(mat_3T_txt)
        combiner = np.array(sp.N(combiner_s, ), dtype=np.complex128)

        kmat = np.array([[0.0, 1.0, -1.0], ])

        include_iotags = True
        include_downsampling = True
        if include_iotags:
            from astropy.table import Column

            outbright = data = np.array([True, False, False])[None, :]
            outphot = data = np.array([False, False, False])[None, :]
            outdark = data = np.array([False, True, True])[None, :]
            inpol = data = np.array(["s", "s", "s"])[None, :]
            outpol = data = np.array(["s", "s", "s"])[None, :]

        # collector positions
        baseline = 15  # in meter
        # Collector diameter
        telescope_diam = 3.0

        # rotation angles over observation
        n_sample_time = 100
        rotation_angle = np.linspace(0., 2 * np.pi, n_sample_time)  # in rad

        # collector_positions_init = np.array(((-baseline/2, baseline/2),
        #                                      (0, 0)))

        collector_positions_init = np.array(((-baseline / 2, baseline / 2, 0),
                                             (0, 0, baseline / 2)))

        rotation_matrix = np.array(((np.cos(rotation_angle), -np.sin(rotation_angle)),
                                    (np.sin(rotation_angle), np.cos(rotation_angle))))

        collector_position = np.dot(np.swapaxes(rotation_matrix, -1, 0), collector_positions_init)

        # observing wavelengths
        n_wl_bin = 5
        wl_bins = np.linspace(4.0e-6, 18.0e-6, n_wl_bin)  # in meter

        # collector area
        scaled_area = 1  # in meter^2

        # Measurement covariance
        # np.random.seed(10)
        # np.random.normal(loc=(), size=wl_bin.shape)
        cov = 1e1 * np.eye(kmat.shape[0] * wl_bins.shape[0])
        covs = np.array([cov for i in range(n_sample_time)])

        ####################################################################
        collector_positions_init.T
        from astropy.table import Table

        myarraytable = Table(names=["TEL_NAME", "STA_NAME", "STA_INDEX", "DIAMETER", "STAXYZ"],
                             dtype=[str, str, int, float, "(3,)double"],
                             units=[None, None, None, "m", "m"])
        for i, (atelx, ately) in enumerate(collector_positions_init.T):
            myarraytable.add_row([f"Tel {i}", f"", i, telescope_diam, np.array([atelx, ately, 0.])])

        ####################################################################
        oi_array = io.OI_ARRAY(data_table=myarraytable, header=io.OI_ARRAY_DEFAULT_VLTI_HEADER)
        ni_catm = io.NI_CATM(data_array=combiner[None, :, :] * np.ones_like(wl_bins)[:, None, None])
        mykmat = io.NI_KMAT(data_array=kmat)
        from copy import copy

        my_FOV_header = copy(io.NI_FOV_DEFAULT_HEADER)
        my_FOV_header["NIFITS FOV_TELDIAM"] = telescope_diam
        my_FOV_header["NIFITS FOV_TELDIAM_UNIT"] = "m"
        ni_fov = io.NI_FOV.simple_from_header(header=my_FOV_header, lamb=wl_bins,
                                              n=n_sample_time)

        oi_target = io.OI_TARGET.from_scratch()
        oi_target.add_target(target='Test Target',
                             raep0=14.3,
                             decep0=-60.4)

        mykcov = ni_kcov = io.NI_KCOV(data_array=covs, unit=(u.ph / u.s))

        from astropy.table import Table, Column
        from astropy.time import Time

        n_telescopes = combiner.shape[1]
        total_obs_time = 10 * 3600  # s
        times_relative = np.linspace(0, total_obs_time, n_sample_time)
        warnings.simplefilter('ignore', ErfaWarning)  # to suppress warnings from astropy.time in the next line
        dateobs = Time("2035-06-23T00:00:00.000") + times_relative * u.s
        mjds = dateobs.to_value("mjd")
        seconds = (dateobs - dateobs[0]).to_value("s")
        target_id = np.zeros_like(times_relative)
        app_index = np.arange(n_telescopes)[None, :] * np.ones(n_sample_time)[:, None]
        target_ids = 0 * np.ones(n_sample_time)
        int_times = np.gradient(seconds)
        mod_phas = np.ones((n_sample_time, n_wl_bin, n_telescopes), dtype=complex)
        appxy = collector_position.transpose((0, 2, 1))
        arrcol = np.ones((n_sample_time, n_telescopes)) * np.pi * telescope_diam ** 2 / 4
        fov_index = np.ones(n_sample_time)

        app_index = Column(data=app_index, name="APP_INDEX",
                           unit=None, dtype=int)
        target_id = Column(data=target_ids, name="TARGET_ID",
                           unit=None, dtype=int)
        times_relative = Column(data=seconds, name="TIME",
                                unit="", dtype=float)
        mjds = Column(data=mjds, name="MJD",
                      unit="day", dtype=float)
        int_times = Column(data=seconds, name="INT_TIME",
                           unit="s", dtype=float)
        mod_phas = Column(data=mod_phas, name="MOD_PHAS",
                          unit=None, dtype=complex)
        appxy = Column(data=appxy, name="APPXY",
                       unit="m", dtype=float)
        arrcol = Column(data=arrcol, name="ARRCOL",
                        unit="m^2", dtype=float)
        fov_index = Column(data=fov_index, name="FOV_INDEX",
                           unit=None, dtype=int)
        mymod_table = Table()
        mymod_table.add_columns((app_index, target_id, times_relative, mjds,
                                 int_times, mod_phas, appxy, arrcol, fov_index))
        mymod_table
        mynimod = io.NI_MOD(mymod_table)

        ###########################################

        from astropy.io import fits

        wl_data = np.hstack((wl_bins[:, None], np.gradient(wl_bins)[:, None]))
        wl_table = Table(data=wl_data, names=("EFF_WAVE", "EFF_BAND"), dtype=(float, float))
        wl_table

        del wl_data
        oi_wavelength = io.OI_WAVELENGTH(data_table=wl_table, )
        # oi_wavelength = io.OI_WAVELENGTH()

        if include_downsampling:
            ni_oswavelength = io.NI_OSWAVELENGTH(data_table=wl_table, )
            ni_dsamp = io.NI_DSAMP(data_array=np.eye(len(wl_table)))
        else:
            ni_oswavelength = None
            ni_dsamp = None

        if include_iotags:

            outbrightcol = Column(data=outbright,
                                  name="BRIGHT", unit=None, dtype=bool)
            outphotcol = Column(data=outphot,
                                name="PHOT", unit=None, dtype=bool)
            outdarkcol = Column(data=outdark,
                                name="DARK", unit=None, dtype=str)
            inpolcol = Column(data=inpol,
                              name="OUTPOLA", unit=None, dtype=str)
            outpolcol = Column(data=outpol,
                               name="INPOLA", unit=None, dtype=str)
            iotags_table = Table()
            iotags_table.add_columns((outbrightcol, outphotcol, outdarkcol, inpolcol, outpolcol))
            ni_iotags = io.NI_IOTAGS(data_table=iotags_table)
        else:
            ni_iotags = None

        myheader = fits.Header()
        self.nifits = io.nifits(header=myheader,
                                ni_catm=ni_catm,
                                ni_fov=ni_fov,
                                oi_target=oi_target,
                                oi_wavelength=oi_wavelength,
                                ni_mod=mynimod,
                                ni_kmat=mykmat,
                                ni_kcov=mykcov,
                                ni_dsamp=ni_dsamp,
                                ni_oswavelength=ni_oswavelength,
                                ni_iotags=ni_iotags)
