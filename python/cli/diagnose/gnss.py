import numpy as np

class Ellipsoid:
    def __init__(self, a, b):
        self.a = a # semi-major axis
        self.b = b # semi-minor axis
        f = 1.0 - b / a # flattening factor
        self.e2 = 2 * f - f ** 2 # eccentricity squared

class GnssConverter:
    def __init__(self):
        self.ell = Ellipsoid(a=6378137.0, b=6356752.31424518) # WGS-84 ellipsoid
        self.initialized = False
        self.originECEF = None
        self.R_ecef2enu = None
        self.R_enu2ecef = None
        self.prev = {"x": 0, "y": 0, "z": 0}

    def set_origin(self, lat, lon, alt):
        def ecef_to_enu_rotation_matrix(lat, lon):
            lat = np.deg2rad(lat)
            lon = np.deg2rad(lon)

            return np.array([
                [-np.sin(lon), np.cos(lon), 0],
                [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
            ])

        self.initialized = True
        self.originECEF = self.__geodetic2ecef(lat, lon, alt)
        self.R_ecef2enu = ecef_to_enu_rotation_matrix(lat, lon)
        self.R_enu2ecef = self.R_ecef2enu.T

    def __geodetic2ecef(self, lat, lon, alt):
        # https://gssc.esa.int/navipedia/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
        a = self.ell.a
        e2 = self.ell.e2
        N = a / np.sqrt(1 - e2 * np.sin(lat) * np.sin(lat)) # radius of curvature in the prime vertical

        x = (N + alt) * np.cos(lat) * np.cos(lon)
        y = (N + alt) * np.cos(lat) * np.sin(lon)
        z = ((1 - e2) * N + alt) * np.sin(lat)
        return np.array([x, y, z])

    def __ecef2geodetic(self, x, y, z):
        # https://gssc.esa.int/navipedia/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion
        a = self.ell.a
        e2 = self.ell.e2
        p = np.sqrt(x**2 + y**2)
        lon = np.arctan2(y, x)

        # latitude and altitude are computed by an iterative procedure.
        MAX_ITERS = 1000
        MIN_LATITUDE_CHANGE_RADIANS = 1e-10
        MIN_ALTITUDE_CHANGE_METERS = 1e-6
        lat_prev = np.arctan(z / ((1-e2)*p)) # initial value
        alt_prev = -100000 # arbitrary
        for _ in range(MAX_ITERS):
            N_i = a / np.sqrt(1-e2*np.sin(lat_prev)**2)
            alt_i = p / np.cos(lat_prev) - N_i
            lat_i = np.arctan(z / ((1 - e2 * (N_i/(N_i + alt_i)))*p))
            if abs(lat_i - lat_prev) < MIN_LATITUDE_CHANGE_RADIANS and abs(alt_i - alt_prev) < MIN_ALTITUDE_CHANGE_METERS: break
            alt_prev = alt_i
            lat_prev = lat_i

        lat = np.rad2deg(lat_i)
        lon = np.rad2deg(lon)
        return np.array([lat, lon, alt_i])

    def __ecef2enu(self, x, y, z):
        # https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
        assert(self.initialized)
        xyz = np.array([x, y, z])
        xyz = xyz - self.originECEF
        return self.R_ecef2enu @ xyz

    def __enu2ecef(self, e, n, u):
        # https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
        assert(self.initialized)
        enu = np.array([e, n, u])
        xyz = self.R_enu2ecef @ enu
        return xyz + self.originECEF

    def enu(self, lat, lon, alt=0, accuracy=1.0, minAccuracy=-1.0):
         # Filter out inaccurate measurements to make pose alignment easier.
        if (minAccuracy > 0.0 and (accuracy > minAccuracy or accuracy < 0.0)):
            return self.prev

        if not self.initialized:
            self.set_origin(lat, lon, alt)

        x, y, z = self.__geodetic2ecef(lat, lon, alt)
        enu = self.__ecef2enu(x, y, z)
        enu = { "x": enu[0], "y": enu[1], "z": enu[2] }
        self.prev = enu
        return enu

    def wgs(self, e, n, u):
        assert(self.initialized)
        x, y, z = self.__enu2ecef(e, n, u)
        wgs = self.__ecef2geodetic(x, y, z)
        return { "latitude": wgs[0], "longitude": wgs[1], "altitude": wgs[2] }

    def wgs_array(self, pos):
        assert(self.initialized)
        arr = []
        for enu in pos:
            arr.append(self.wgs(enu[0], enu[1], enu[2]))
        return arr