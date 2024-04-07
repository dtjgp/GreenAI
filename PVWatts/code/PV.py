# the code is to create a class for solar panel
# the class has the following attributes:
# 1. the location of the solar panel
# 2. the latitude of the solar panel
# 3. the longitude of the solar panel
# 4. the elevation of the solar panel
# 5. the DC system size of the solar panel
# 6. the type of the solar panel
# 7. the array type of the solar panel
# 8. the array tilt angle of the solar panel
# 9. the array azimuth angle of the solar panel
# 10. the system loss of the solar panel
# 11. the DC to AC size ratio of the solar panel
# 12. the inverter efficiency of the solar panel
# 13. the ground coverage ratio of the solar panel
# 14. the albedo of the solar panel
# 15. the bifaciality of the solar panel

class PV:
    def __init__(self, location, latitude, longitude, elevation, DC_size, panel_type, array_type, tilt_angle, azimuth_angle, system_loss, DC_AC_ratio, inverter_efficiency, ground_coverage_ratio, albedo, bifaciality):
        self.location = location
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.DC_size = DC_size
        self.panel_type = panel_type
        self.array_type = array_type
        self.tilt_angle = tilt_angle
        self.azimuth_angle = azimuth_angle
        self.system_loss = system_loss
        self.DC_AC_ratio = DC_AC_ratio
        self.inverter_efficiency = inverter_efficiency
        self.ground_coverage_ratio = ground_coverage_ratio
        self.albedo = albedo
        self.bifaciality = bifaciality
        
        