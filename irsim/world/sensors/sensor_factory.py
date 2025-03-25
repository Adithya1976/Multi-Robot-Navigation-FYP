from irsim.world.sensors.lidar2d import Lidar2D
from irsim.world.sensors.lidar2d_custom import CustomLidar2D


class SensorFactory:

    def create_sensor(self, state, obj_id, external_objects, **kwargs):
        
        sensor_type = kwargs.get("name", kwargs.get("type", "lidar2d"))

        if sensor_type == "lidar2d":
            return Lidar2D(state, obj_id, **kwargs)
        elif sensor_type == "lidar2d_custom":
            return CustomLidar2D(state, obj_id, external_objects=external_objects, **kwargs)
        else:
            raise NotImplementedError(f"Sensor types {type} not implemented")

