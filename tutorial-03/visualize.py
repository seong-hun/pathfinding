import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


class Object:
    def __init__(self):
        self.base_vertices = {}
        self.geometries = []

    def register(self, *geometries):
        for geometry in geometries:
            self.base_vertices[geometry] = np.asarray(geometry.vertices).copy()
            self.geometries.append(geometry)

    def rotate(self, rotation_matrix):
        for geometry in self.geometries:
            vertices = self.base_vertices[geometry] @ rotation_matrix.T
            geometry.vertices = o3d.utility.Vector3dVector(vertices)


# XYZ axes
def make_arrow(angles, color):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.01,
        cone_radius=0.03,
        cylinder_height=1.7,
        cone_height=0.06,
    )
    R = Rotation.from_euler("ZYX", angles).as_matrix()
    arrow.rotate(R, center=(0, 0, 0))
    arrow.paint_uniform_color(color)
    return arrow


class Aircraft(Object):
    _num = 0

    def __init__(self, color=[1, 0.706, 0]):
        super().__init__()
        self.num = Aircraft._num = Aircraft._num + 1

        # Aircrat mesh
        mesh = o3d.io.read_triangle_mesh("aircraft.ply")
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)

        arrows = [
            make_arrow(angles, color=color)
            for angles in [
                [0, np.pi / 2, 0],
                [0, 0, -np.pi / 2],
                [0, 0, 0],
            ]
        ]

        self.register(*arrows)
        self.register(mesh)

    def __repr__(self):
        return f"<Aircraft {self.num}>"


class Camera:
    params = {
        "field_of_view": 60.0,
        "front": [
            0.62063766017086752,
            0.56878270344099702,
            -0.53971764010821588,
        ],
        "lookat": [
            0.1914881525332742,
            0.084610172166532507,
            0.21062106320062166,
        ],
        "up": [
            -0.42345526140744039,
            -0.3361815994393404,
            -0.84122979844080847,
        ],
        "zoom": 0.69999999999999996,
    }

    def __init__(self, vis):
        self.ctr = vis.get_view_control()

    def set(self):
        self.ctr.set_lookat(self.params["lookat"])
        self.ctr.set_up(self.params["up"])
        self.ctr.set_front(self.params["front"])
        self.ctr.set_zoom(self.params["zoom"])


class EulerScaler:
    def __init__(self, rot):
        self.angles = rot.as_euler("ZYX")

    def __repr__(self):
        return "<EulerScaler: {:5.2f}, {:5.2f}, {:5.2f}>".format(
            *np.rad2deg(self.angles)
        )

    def scale(self, factor):
        return Rotation.from_euler("ZYX", self.angles * factor)


class Animator:
    def __init__(self, *objs):
        self.objs = objs

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        for obj in self.objs:
            for geometry in obj.geometries:
                self.vis.add_geometry(geometry)

        self.camera = Camera(self.vis)
        self.rotation_list = []

    def add_euler(self, obj, rot):
        angles = rot.as_euler("ZYX")

        angle_vectors = []
        for i, angle in enumerate(angles):
            if np.isclose(angle, 0):
                continue
            vec = np.zeros(3)
            vec[i] = angle
            angle_vectors.append(vec)

        for angle_vector in angle_vectors:
            self.rotation_list.append(
                (obj, EulerScaler(Rotation.from_euler("ZYX", angle_vector)))
            )

    def make_frames(self):
        prev_rotations = {obj: Rotation.identity() for obj in self.objs}

        for obj, rotation in self.rotation_list:
            prev_rotation = prev_rotations[obj]
            current_rotation = Rotation.identity()

            for scale in np.linspace(0, 1, 100):
                current_rotation = prev_rotation * rotation.scale(scale)
                yield obj, current_rotation

            prev_rotations[obj] = current_rotation

    def show(self):
        self.camera.set()

        frames = self.make_frames()

        for obj, rot in frames:
            obj.rotate(rot.as_matrix())

            for geometry in obj.geometries:
                self.vis.update_geometry(geometry)

            self.vis.poll_events()
            self.vis.update_renderer()

        self.vis.run()
        self.vis.destroy_window()


def show(*rotations):
    plant1 = Aircraft(color=[0.5, 0.5, 0.5])
    plant2 = Aircraft(color=[0.3, 0.7, 0.3])

    animator = Animator(plant1, plant2)

    for rotation in rotations:
        animator.add_euler(plant2, rotation)

    animator.show()


def main():
    R0 = Rotation.from_euler("ZYX", [45, 30, 15], degrees=True)
    R1 = Rotation.from_euler("XYZ", [-15, -30, -45], degrees=True)
    show(R0, R1)


if __name__ == "__main__":
    main()
