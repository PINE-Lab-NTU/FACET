"""Optional RealSense sources with explicit role mapping; no machine serials."""

import numpy as np


class RealSenseCameras:
    def __init__(self, config):
        import pyrealsense2 as rs

        self.rs, self.config, self.pipelines = rs, config, {}
        roles = {"view1", "hand", "view2"}
        if set(config.serials) & set(config.aliases) or set(config.serials) | set(config.aliases) != roles:
            raise ValueError("assign every camera role once, by serial or alias")
        if len(set(config.serials.values())) != len(config.serials):
            raise ValueError("use camera aliases to share a physical device")
        if any(v not in config.serials for v in config.aliases.values()):
            raise ValueError("camera aliases must name physical roles")
        try:
            for role, serial in config.serials.items():
                pipeline, setup = rs.pipeline(), rs.config()
                setup.enable_device(serial)
                setup.enable_stream(rs.stream.color, config.width, config.height, rs.format.rgb8, config.fps)
                pipeline.start(setup)
                self.pipelines[role] = pipeline
        except Exception:
            self.close()
            raise

    def capture(self):
        frames = {}
        for role, pipeline in self.pipelines.items():
            frame = pipeline.wait_for_frames(timeout_ms=500).get_color_frame()
            if not frame:
                raise RuntimeError(f"missing camera frame: {role}")
            frames[role] = np.asanyarray(frame.get_data()).copy()
        for role, source in self.config.aliases.items():
            frames[role] = frames[source].copy()
        return frames

    def close(self):
        for p in self.pipelines.values():
            p.stop()
        self.pipelines.clear()


class MockCameras:
    def capture(self):
        return {r: np.zeros((32, 32, 3), np.uint8) for r in ("view1", "hand", "view2")}

    def close(self):
        pass
