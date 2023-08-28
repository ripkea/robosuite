import mujoco as mj
import numpy as np
import random
import math
import cv2 as cv
import mujoco.viewer as view
import time
from scipy.spatial.transform import Rotation
import transforms3d.quaternions as tq

field_size = 128
base_height = 20
height_perturbation_range = 5
angle_of_repose_deg = 38
angle_of_repose_rad = angle_of_repose_deg * math.pi / 180
c = (1/math.sin((90-angle_of_repose_deg)* math.pi / 180))
critical_height = math.sin(angle_of_repose_rad) * c
flow_rate = 0.125
random.seed(42)
MOVE_SPEED = 0.006
last_tool_grid = None
x_coords = np.indices((field_size, field_size))[0]
y_coords = np.rot90(x_coords)
heightmap_coords = np.stack((x_coords, y_coords), axis=2)
fragment_counter = 0
tool_body_index = 19
fragment_body_index = 1
fragment_force_threshold = 0.005

grid_to_world_factor = 0.0005

kernel = np.ones((3, 3))

half_extent_tool_x = .01
half_extent_tool_y = .002
half_extent_tool_z = .01

class SandSimulator:

    def __init__(self, model, data):
        self.heightfield = None
        self.model = model
        self.data = data
        
    def get_scaled_heightfield(self):
        return scale_hfield_data_before_upload(self.heightfield)

    def scale_hfield_data_before_upload(hfield_data):
        return np.copy(hfield_data)*0.01


    def get_line_equation(vector, starting_point):
        if vector[0] != 0:
            m = vector[1] / vector[0]
        else:
            m = float('inf')

        b = starting_point[1] - m * starting_point[0]
        return m,b


    def get_side_of_line(m, b, rectangle_center, movement_vec):
        if m == float('inf'):
            result = np.zeros_like(x_coords)
            if abs(movement_vec[0]) > abs(movement_vec[1]):
                result[:, int(rectangle_center[0]):] = 1.0
            else:
                result[:, :int(rectangle_center[1])] = 1.0
            return result
        vector_x = y_coords
        vector_y = x_coords
        y_prime = m*vector_x+b
        result = vector_y - y_prime
        result = result.clip(0,1)
        result[result > 0.] = 1.
        return np.array(result)


    def get_perp_vec(vec):
        return [vec[1], -1.*vec[0]]


    def get_tool_coords():
        # Get rectangle coordinates (world and grid)
        face_vertices_local = [
            [-half_extent_tool_x, -half_extent_tool_y, -half_extent_tool_z],
            [half_extent_tool_x, -half_extent_tool_y, -half_extent_tool_z],
            [half_extent_tool_x, -half_extent_tool_y, half_extent_tool_z],
            [half_extent_tool_x, half_extent_tool_y, -half_extent_tool_z],
            [-half_extent_tool_x, half_extent_tool_y, half_extent_tool_z]
        ]

        # Get the base position and orientation of the box
        base_pos = get_geom_center('tool')
        base_ori_quat = model.geom('tool').quat
        base_rot = np.array(Rotation.from_quat(base_ori_quat).as_matrix()).reshape(3, 3)

        # Convert the local face vertices to world coordinates
        rectangle_down_vertices = []
        for vertex_local in face_vertices_local:
            vertex_world = base_pos + np.dot(base_rot, vertex_local)
            vertex_world = [vertex_world[0], vertex_world[1], vertex_world[2]]
            rectangle_down_vertices.append(vertex_world)

        rectangle_grid = np.array(rectangle_down_vertices)*10+np.array([field_size/2, field_size/2,0]).astype(int)

        return rectangle_grid


    def displace_sand_for_tool(heightfield_copy):
        global last_tool_grid
        updated_heights = get_heights_by_ray_cast('tool')
        overlap = updated_heights - heightfield_copy
        occupancy_mask = np.clip(np.where(overlap <= 0.0, 1.0, 0.), 0., 1.)
        sand_to_displace = np.clip(overlap, -5000, 0.)

        tool_grid = get_tool_coords()
        tool_center_grid = (tool_grid[0]+tool_grid[3])/2

        while np.min(sand_to_displace) < 0.0:
            reduced_heightfield = heightfield_copy + sand_to_displace
            displaced_amount = np.sum(heightfield_copy - reduced_heightfield)

            if last_tool_grid is not None:
                last_tool_center_grid = (last_tool_grid[0] + last_tool_grid[3]) / 2
                movement_vec = tool_center_grid - last_tool_center_grid
                perp_movement_vec = get_perp_vec(movement_vec)
                m, b = get_line_equation(perp_movement_vec, tool_center_grid)

            displaced_mask = np.array(
                cv.dilate(occupancy_mask, kernel, iterations=2) - occupancy_mask)
            possible_displacement_mask = get_side_of_line(m, b, tool_center_grid, movement_vec)

            if movement_vec[0] < 0 or movement_vec[1] < 0:
                possible_displacement_mask = 1 - possible_displacement_mask

            if np.sum(abs(movement_vec[:2])) < 0.1:
                possible_displacement_mask = np.ones_like(heightfield_copy)
                edge_movement = last_tool_grid[:, :2] - tool_grid[:, :2]
                edge_a = edge_movement[0]
                edge_b = edge_movement[1]

                if abs(np.sum(edge_a)) > 0.01 or abs(np.sum(edge_b)) > 0.01:
                    long_or_vec = tool_grid[0, :2] - tool_grid[1, :2]
                    short_or_vec = tool_grid[0, :2] - tool_grid[3, :2]
                    m_long, b_long = get_line_equation(long_or_vec, tool_center_grid)
                    m_short, b_short = get_line_equation(short_or_vec, tool_center_grid)
                    long_partition = get_side_of_line(m_long, b_long, tool_center_grid, movement_vec)
                    short_partition = get_side_of_line(m_short, b_short, tool_center_grid, movement_vec)
                    total_partition = long_partition + short_partition
                    total_partition[total_partition != 1] = 0.
                    possible_displacement_mask = total_partition

            displaced_mask *= possible_displacement_mask
            if np.sum(displaced_mask != 0):
                displacement_increment = displaced_mask * (displaced_amount / np.sum(displaced_mask))
                heightfield_copy = reduced_heightfield
                heightfield_copy = heightfield_copy + displacement_increment

            overlap = updated_heights - heightfield_copy
            occupancy_mask = np.clip(np.where(overlap <= 0.0, 1.0, 0.), 0., 1.)
            sand_to_displace = np.clip(overlap, -5000, 0.)

        occupancy_mask[0, :] = 1
        occupancy_mask[:, 0] = 1
        occupancy_mask[field_size - 1, :] = 1
        occupancy_mask[:, field_size - 1] = 1
        last_tool_grid = tool_grid
        return occupancy_mask, heightfield_copy


    def test(heightfield_copy):
        updated_heights = get_heights_by_ray_cast('frag')
        height_mask = np.where(updated_heights < 100, 1., 0.).astype(bool)
        heightfield_copy[height_mask] += 0.1
        return heightfield_copy


    def displace_sand(heightfield_copy):
        global fragment_counter
        geom_id_frag = data.geom('frag').id
        geom_id_sand = data.geom('sand').id

        contact_geoms_1 = data.contact.geom1
        contact_geoms_2 = data.contact.geom2

        indices_to_ignore_1 = np.where((contact_geoms_1 != geom_id_frag) & (contact_geoms_1 != geom_id_sand))
        indices_to_ignore_2 = np.where((contact_geoms_2 != geom_id_frag) & (contact_geoms_2 != geom_id_sand))

        mask_1 = np.ones(len(contact_geoms_1), dtype=bool)
        mask_2 = np.ones(len(contact_geoms_2), dtype=bool)
        mask_1[indices_to_ignore_1] = False
        mask_2[indices_to_ignore_2] = False

        mask = mask_1 & mask_2

        contact_points_grid = convert_world_to_grid(data.contact.pos)
        contact_points_grid = contact_points_grid[mask]

        updated_heights = get_heights_by_ray_cast('frag')
        height_mask = np.where(updated_heights < 100, 1., 0.)
        x = (contact_points_grid[:, 0] + field_size / 2).astype(int)
        y = (contact_points_grid[:, 1] + field_size / 2).astype(int)
        sand_to_displace = np.zeros_like(heightfield_copy)
        overlap = (updated_heights - heightfield_copy)
        occupancy_mask = np.clip(np.where(overlap <= 0.0, 1.0, 0.), 0., 1.)
        occupancy_mask[y,x] = 1.0

        if len(contact_points_grid) > 0 and fragment_counter < 4:
            A = np.array([1, 2, 1])
            B = np.array([2, 3, 4])
            result = np.column_stack((x, y))
            all_contact_forces = np.zeros(6, np.float64)
            for idx in range(len(contact_points_grid)):
                force = np.zeros(6, np.float64)
                mj.mj_contactForce(model, data, idx, force)
                all_contact_forces += force
            avg_contact_force = all_contact_forces / len(contact_points_grid)
            if avg_contact_force[0] < fragment_force_threshold:
                fragment_counter += 1
            else:
                fragment_counter = 0
    #        try:
    #            hull = ConvexHull(result)
    #            area = hull.volume
    #        except:
    #            print("except")
            sand_to_displace[y, x] = -0.75
            occupancy_mask[y,x] = 1.0
            while np.min(sand_to_displace) < 0.0:
                reduced_heightfield = heightfield_copy + sand_to_displace
                displaced_amount = np.sum(heightfield_copy - reduced_heightfield)

                dilated_1_mask = np.array(
                    cv.dilate(height_mask, kernel, iterations=1))
                displaced_mask = np.array(
                    cv.dilate(height_mask, kernel, iterations=2) - dilated_1_mask).clip(0.,1.)

                displaced_mask = displaced_mask.T

                if np.sum(displaced_mask != 0):
                    displacement_increment = displaced_mask * (displaced_amount / np.sum(displaced_mask))
                    heightfield_copy = reduced_heightfield
                    heightfield_copy = heightfield_copy + displacement_increment

                overlap = (updated_heights - heightfield_copy)
                occupancy_mask = np.clip(np.where(overlap <= 0.0, 1.0, 0.), 0., 1.)
                occupancy_mask[y, x] = 1.0
                sand_to_displace = np.clip(overlap, -5000, 0.)

        occupancy_mask[0, :] = 1
        occupancy_mask[:, 0] = 1
        occupancy_mask[field_size - 1, :] = 1
        occupancy_mask[:, field_size - 1] = 1
        return occupancy_mask, heightfield_copy


    def get_heights_by_ray_cast(geom_name):
        ray_results, rays_from = do_bounding_box_ray_cast(geom_name)
        frag_heights = np.zeros((field_size, field_size)) + 100

        if len(ray_results) > 0:

            no_hit_idx = np.argwhere(np.array(ray_results) < 0.)
            result_hit_points = np.copy(rays_from)
            result_hit_points[:, 2] = ray_results

            result_hit_points = np.delete(result_hit_points, no_hit_idx, axis=0)

            if len(result_hit_points) > 0:
                x = (result_hit_points[:, 0] * 10).astype(int) + int(field_size / 2)-1
                y = (result_hit_points[:, 1] * 10).astype(int) + int(field_size / 2)-1
                z = np.array(result_hit_points)[:, 2] * 10

                indices_to_remove = []
                for idx in range(0,len(x)):
                    if x[idx] >= field_size or y[idx] >= field_size or x[idx] < 0 or y[idx]< 0:
                        indices_to_remove.append(idx)
                np.delete(x, indices_to_remove)
                np.delete(y, indices_to_remove)
                np.delete(z, indices_to_remove)
                frag_heights[y, x] = z

        return frag_heights


    def generate_aabb(rotation_quaternion, center_point, half_extents):
        # Convert the rotation quaternion to a rotation matrix
        rotation_matrix = tq.quat2mat(rotation_quaternion)

        # Compute the minimum and maximum corner points of the AABB
        min_point = center_point - np.abs(rotation_matrix @ half_extents)
        max_point = center_point + np.abs(rotation_matrix @ half_extents)

        return min_point, max_point

    def get_geom_center(geom_name):
        geom_id = data.geom(geom_name).id
        geom_position = data.geom(geom_name).xpos
        frag_bounding_box = model.geom_aabb[geom_id]
        center_aabb = np.array([geom_position[0], geom_position[1], geom_position[2]]) + [frag_bounding_box[0],
                                                                                          frag_bounding_box[1],
                                                                                          frag_bounding_box[2]]
        return center_aabb


    def get_tool_pos_correction():
        geom_id = data.geom('tool').id
        frag_bounding_box = model.geom_aabb[geom_id]
        center_aabb = get_geom_center('tool')
        half_extents_aabb = np.array([frag_bounding_box[5], frag_bounding_box[4], frag_bounding_box[3]])
        min_aabb, max_aabb = generate_aabb(model.body_quat[tool_body_index], center_aabb, half_extents_aabb)
        min_overshoot = [-field_size*0.05, -field_size*0.05] - min_aabb[:2]
        max_overshoot = [field_size*0.05, field_size*0.05] - max_aabb[:2]
        pos_correction = [0,0,0]
        if np.max(min_overshoot) > 0:
            pos_correction[:2] = np.clip(min_overshoot,0,5000)
        elif np.min(max_overshoot) < 0:
            pos_correction[:2] = np.clip(max_overshoot,-5000,0)
        return pos_correction


    def do_bounding_box_ray_cast(geom_name):
        geom_id = data.geom(geom_name).id
        frag_bounding_box = model.geom_aabb[geom_id]
        center_aabb = get_geom_center(geom_name)
        half_extents_aabb = np.array([frag_bounding_box[5], frag_bounding_box[4], frag_bounding_box[3]])
        body_index_to_use = fragment_body_index if geom_name == 'frag' else tool_body_index
        min_aabb, max_aabb = generate_aabb(model.body_quat[body_index_to_use], center_aabb, half_extents_aabb)
        min_aabb -= 0.2
        max_aabb += 0.2

        if min_aabb[0] >= field_size*0.05 or min_aabb[1] >= field_size*0.05 or max_aabb[0] <= -field_size*0.05 or max_aabb[0] <= -field_size*0.05:
            return [],[]

        min_aabb = [max(min_aabb[0], -field_size*0.05), max(min_aabb[1], -field_size*0.05)]
        max_aabb = [min(max_aabb[0], field_size*0.05), min(max_aabb[1], field_size*0.05)]

        x_vals = np.arange(round(min_aabb[0], 1), round(max_aabb[0], 1) + 0.1, 0.1)
        y_vals = np.arange(round(min_aabb[1], 1), round(max_aabb[1], 1) + 0.1, 0.1)

        x_y_coords = [[i, j] for i in x_vals for j in y_vals]
        rays_from = np.array([subarray + [0] for subarray in x_y_coords]).astype(float)
        rays_to = np.array([subarray + [100] for subarray in x_y_coords]).astype(float)
        geom_id_arr = np.zeros(1, np.int32)
        all_ray_res = []
        hfield_geom_id = data.geom('sand').id
        for ray_idx in range(len(rays_from)):
            start_point = rays_from[ray_idx]
            geomid = np.zeros(1, np.int32)
            ray_res = mj.mj_ray(
                m=model,
                d=data,
                pnt=start_point,
                vec=[0, 0, 1],
                geomgroup=None,
                flg_static=1,
                bodyexclude=0,
                geomid=geom_id_arr)
            all_ray_res.append(ray_res)
        return all_ray_res, rays_from


    def sink_sand_at_points(coordinates_grid):
        heightfield_copy = np.copy(np.reshape(np.copy(heightfield), (-1, field_size)))
        x = (coordinates_grid[:, 0]+field_size/2).astype(int)
        y = (coordinates_grid[:, 1]+field_size/2).astype(int)

        if len(coordinates_grid) > 0:
            heightfield_copy[y, x] -= 0.2
        return heightfield_copy


    def convert_world_to_grid(coordinates):
        coordinates_grid = coordinates*10
        coordinates_grid = np.around(coordinates_grid)
        return coordinates_grid


    def initialize_heightfield(self):
        self.heightfield = np.ones((field_size, field_size)).flatten()*10

        self.heightfield = change_heightfield(self.heightfield)

        heightfield_copy = np.reshape(np.copy(self.heightfield), (-1, field_size))

        heightfield_copy[0, :] = 1
        heightfield_copy[:, 0] = 1
        heightfield_copy[field_size - 1, :] = 1
        heightfield_copy[:, field_size - 1] = 1

        if spawn_sand_mount:
            heightfield_copy[int(field_size/2)-10:int(field_size/2)+10,int(field_size/2)-10: int(field_size/2)+10] = 30
        self.heightfield = heightfield_copy.flatten()

    def change_heightfield(heightfield):
        for x in range(int(field_size / 2)):
            for y in range(int(field_size / 2)):
                height = random.uniform(base_height, height_perturbation_range + base_height)
                heightfield[2 * y + 2 * x * field_size] = height
                heightfield[2 * y + 1 + 2 * x * field_size] = height
                heightfield[2 * y + (2 * x + 1) * field_size] = height
                heightfield[2 * y + 1 + (2 * x + 1) * field_size] = height

        return heightfield


    def simulate_sand():
        is_unstable = True
        heightfield_grid = np.reshape(np.copy(self.heightfield), (-1, field_size))
        #heightfield_grid = test(heightfield_grid)
        occupancy_mask, heightfield_grid = displace_sand(heightfield_grid)
        occupancy_mask, heightfield_grid = displace_sand_for_tool(heightfield_grid)
        iter = 0
        while is_unstable:
            iter += 1
            delta_h = np.array(get_unstable_cells(heightfield_grid, occupancy_mask)).astype(float)
            is_unstable = np.sum(np.array(abs(delta_h))) > 0.1
            heightfield_grid = heightfield_grid + delta_h

        #(np.sum(heightfield_grid))
        return heightfield_grid.flatten()


    def get_boundary_occupancy_mask():
        occupancy_mask = np.zeros((field_size, field_size))
        occupancy_mask[0, :] = 1
        occupancy_mask[:, 0] = 1
        occupancy_mask[field_size - 1, :] = 1
        occupancy_mask[:, field_size - 1] = 1
        return occupancy_mask


    def get_unstable_cells(heightfield_copy, occupancy_mask):
        north = np.roll(heightfield_copy, -1, axis=0)
        northeast = np.roll(north, 1, axis=1)
        northwest = np.roll(north, -1, axis=1)

        south = np.roll(heightfield_copy, 1, axis=0)
        southeast = np.roll(south, 1, axis=1)
        southwest = np.roll(south, -1, axis=1)

        west = np.roll(heightfield_copy, -1, axis=1)
        east = np.roll(heightfield_copy, 1, axis=1)

        occ_north = 1 - np.clip(np.roll(occupancy_mask, -1, axis=0) - occupancy_mask, 0., 1.)

        occ_north_east = 1 - np.clip(np.roll(np.roll(occupancy_mask, -1, axis=0), 1, axis=1) - occupancy_mask, 0., 1.)
        occ_north_west = 1 - np.clip(np.roll(np.roll(occupancy_mask, -1, axis=0), -1, axis=1) - occupancy_mask, 0., 1.)

        occ_south = 1 - np.clip(np.roll(occupancy_mask, 1, axis=0) - occupancy_mask, 0., 1.)
        occ_south_east = 1 - np.clip(np.roll(np.roll(occupancy_mask, 1, axis=0), 1, axis=1) - occupancy_mask, 0., 1.)
        occ_south_west = 1 - np.clip(np.roll(np.roll(occupancy_mask, 1, axis=0), -1, axis=1) - occupancy_mask, 0., 1.)

        occ_west = 1 - np.clip(np.roll(occupancy_mask, -1, axis=1) - occupancy_mask, 0., 1.)
        occ_east = 1 - np.clip(np.roll(occupancy_mask, 1, axis=1) - occupancy_mask, 0., 1.)

        # Compute the differences between the cell and its neighbors
        diff_down = (heightfield_copy - north) * occ_north
        diff_down_right = (heightfield_copy - northwest) * occ_north_west
        diff_down_left = (heightfield_copy - northeast) * occ_north_east

        diff_up = (heightfield_copy - south) * occ_south
        diff_up_right = (heightfield_copy - southwest) * occ_south_west
        diff_up_left = (heightfield_copy - southeast) * occ_south_east

        diff_right = (heightfield_copy - west) * occ_west
        diff_left = (heightfield_copy - east) * occ_east

        slope_down = -np.arctan(diff_down)
        slope_down_right = -np.arctan(diff_down_right)
        slope_down_left = -np.arctan(diff_down_left)

        slope_up = -np.arctan(diff_up)
        slope_up_right = -np.arctan(diff_up_right)
        slope_up_left = -np.arctan(diff_up_left)

        slope_right = -np.arctan(diff_right)
        slope_left = -np.arctan(diff_left)

        q_down = get_q(slope_down)
        q_down_right = get_q(slope_down_right)
        q_down_left = get_q(slope_down_left)

        q_up = get_q(slope_up)
        q_up_right = get_q(slope_up_right)
        q_up_left = get_q(slope_up_left)

        q_right = get_q(slope_right)
        q_left = get_q(slope_left)

        delta_h = q_down + q_down_right + q_down_left + q_up + q_up_right + q_up_left + q_right + q_left
        delta_h *= 0.125
        delta_h *= 1 - occupancy_mask
        delta_h *= 180 / math.pi
        return delta_h


    def get_q(input_array):
        q = np.where(input_array < -angle_of_repose_rad, (input_array + angle_of_repose_rad) *    flow_rate, 0)
        q += np.where(input_array > angle_of_repose_rad, (input_array - angle_of_repose_rad) * flow_rate, 0)
        return q

