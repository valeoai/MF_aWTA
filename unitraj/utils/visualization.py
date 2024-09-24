import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# input
# ego: (16,3)
# agents: (16,n,3)
# map: (150,n,3)

# visualize all of the agents and ego in the map, the first dimension is the time step,
# the second dimension is the number of agents, the third dimension is the x,y,theta of the agent
# visualize ego and other in different colors, visualize past and future in different colors,past is the first 4 time steps, future is the last 12 time steps
# visualize the map, the first dimension is the lane number, the second dimension is the x,y,theta of the lane
# you can discard the last dimension of all the elements

def check_loaded_data(data, index=0):
    agents = np.concatenate([data['obj_trajs'][..., :2], data['obj_trajs_future_state'][..., :2]], axis=-2)
    map = data['map_polylines']

    agents = agents[index]
    map = map[index]
    ego_index = data['track_index_to_predict'][index]
    ego_agent = agents[ego_index]

    fig, ax = plt.subplots()

    def draw_line_with_mask(point1, point2, color, line_width=4):
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color)

    def interpolate_color(t, total_t):
        # Start is green, end is blue
        return (0, 1 - t / total_t, t / total_t)

    def interpolate_color_ego(t, total_t):
        # Start is red, end is blue
        return (1 - t / total_t, 0, t / total_t)

    # Function to draw lines with a validity check

    # Plot the map with mask check
    for lane in map:
        if lane[0, -3] in [1, 2, 3]:
            continue
        for i in range(len(lane) - 1):
            if lane[i, -3] > 0:
                draw_line_with_mask(lane[i, :2], lane[i, -2:], color='grey', line_width=1)

    # Function to draw trajectories
    def draw_trajectory(trajectory, line_width, ego=False):
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)
            else:
                color = interpolate_color(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)

    # Draw trajectories for other agents
    for i in range(agents.shape[0]):
        draw_trajectory(agents[i], line_width=2)
    draw_trajectory(ego_agent, line_width=2, ego=True)
    # Set labels, limits, and other properties
    vis_range = 100
    # ax.legend()
    ax.set_xlim(-vis_range + 30, vis_range + 30)
    ax.set_ylim(-vis_range, vis_range)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    # As defined in the common_utils.py file
    # traj_type = { 0: "stationary", 1: "straight", 2: "straight_right",
    #         3: "straight_left", 4: "right_u_turn", 5: "right_turn",
    #         6: "left_u_turn", 7: "left_turn" }
    #
    # kalman_2s, kalman_4s, kalman_6s = list(data["kalman_difficulty"][index])
    #
    # plt.title("%s -- Idx: %d -- Type: %s  -- kalman@(2s,4s,6s): %.1f %.1f %.1f" % (1, index, traj_type[data["trajectory_type"][0]], kalman_2s, kalman_4s, kalman_6s))
    # # Return the axes object
    # plt.show()

    # Return the PIL image
    return plt
    # return ax


def concatenate_images(images, rows, cols):
    # Determine individual image size
    width, height = images[0].size

    # Create a new image with the total size
    total_width = width * cols
    total_height = height * rows
    new_im = Image.new('RGB', (total_width, total_height))

    # Paste each image into the new image
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        new_im.paste(image, (col * width, row * height))

    return new_im


def concatenate_varying(image_list, column_counts):
    if not image_list or not column_counts:
        return None

    # Assume all images have the same size, so we use the first one to calculate ratios
    original_width, original_height = image_list[0].size
    total_height = original_height * column_counts[0]  # Total height is based on the first column

    columns = []  # To store each column of images

    start_idx = 0  # Starting index for slicing image_list

    for count in column_counts:
        # Calculate new height for the current column, maintaining aspect ratio
        new_height = total_height // count
        scale_factor = new_height / original_height
        new_width = int(original_width * scale_factor)

        column_images = []
        for i in range(start_idx, start_idx + count):
            # Resize image proportionally
            resized_image = image_list[i].resize((new_width, new_height), Image.Resampling.LANCZOS)
            column_images.append(resized_image)

        # Update start index for the next batch of images
        start_idx += count

        # Create a column image by vertically stacking the resized images
        column = Image.new('RGB', (new_width, total_height))
        y_offset = 0
        for img in column_images:
            column.paste(img, (0, y_offset))
            y_offset += img.height

        columns.append(column)

    # Calculate the total width for the new image
    total_width = sum(column.width for column in columns)

    # Create the final image to concatenate all column images
    final_image = Image.new('RGB', (total_width, total_height))
    x_offset = 0
    for column in columns:
        final_image.paste(column, (x_offset, 0))
        x_offset += column.width

    return final_image


def visualize_prediction(batch, prediction, draw_index=0):
    def draw_line_with_mask(point1, point2, color, line_width=4):
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color)

    def interpolate_color(t, total_t):
        # Start is green, end is blue
        return (0, 1 - t / total_t, t / total_t)

    def interpolate_color_ego(t, total_t):
        # Start is red, end is blue
        return (1 - t / total_t, 0, t / total_t)

    def draw_trajectory(trajectory, line_width, ego=False):
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)
            else:
                color = interpolate_color(t, total_t)
                color = (0,1,0,1) # green
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)

    batch = batch['input_dict']
    map_lanes = batch['map_polylines'][draw_index].cpu().numpy()
    map_mask = batch['map_polylines_mask'][draw_index].cpu().numpy()
    past_traj = batch['obj_trajs'][draw_index].cpu().numpy()
    future_traj = batch['obj_trajs_future_state'][draw_index].cpu().numpy()
    past_traj_mask = batch['obj_trajs_mask'][draw_index].cpu().numpy()
    future_traj_mask = batch['obj_trajs_future_mask'][draw_index].cpu().numpy()
    pred_future_prob = prediction['predicted_probability'][draw_index].detach().cpu().numpy()
    pred_future_traj = prediction['predicted_trajectory'][draw_index].detach().cpu().numpy()
    
    pred_center_gt_trajs = batch['center_gt_trajs'][draw_index,:,:2].cpu().numpy()
    #print('pred_center_gt_trajs.shape', pred_center_gt_trajs.shape)
    map_xy = map_lanes[..., :2]

    map_type = map_lanes[..., 0, -20:]
    #print("pred_future_prob.shape ", pred_future_prob.shape)
    # draw map
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # Plot the map with mask check
    for idx, lane in enumerate(map_xy):
        lane_type = map_type[idx]
        # convert onehot to index
        lane_type = np.argmax(lane_type)
        if lane_type in [1, 2, 3]:
            continue
        for i in range(len(lane) - 1):
            if map_mask[idx, i] and map_mask[idx, i + 1]:
                draw_line_with_mask(lane[i], lane[i + 1], color='grey', line_width=1.5)

    # draw past trajectory
    #for idx, traj in enumerate(past_traj):
    #    draw_trajectory(traj, line_width=2)
    #print("future_traj.shape ", future_traj.shape)
    #print()
    ## draw future trajectory
    #for idx, traj in enumerate(future_traj):
    #    draw_trajectory(traj, line_width=2)

    # predicted future trajectory is (n,future_len,2) with n possible future trajectories, visualize all of them 
    colors = [(1, 0.75294118, 0.79607843,1.0), (1.0, 0.41176471, 0.70588235, 1.0), (1.0, 0.07843137, 0.57647059,1.0) , (0.85882353, 0.43921569, 0.57647059, 1.0), (0.78039216, 0.08235294, 0.52156863,1.0),(1,0,0,1.0)]
    colors = colors[::-1]
    sorted_prob_idx = np.argsort(pred_future_prob)[::-1][:64].tolist() #top 10 highest scores
    
    #print(pred_future_prob[sorted_prob_idx])
    #print(past_traj_mask.sum(-1))
    
    #for idx, traj in enumerate(pred_future_traj):
    #    # calculate color based on probability
    #    #color = cm.hot(pred_future_prob[idx])
    #    if idx in sorted_prob_idx:
    #        #try:
    #        color = colors[sorted_prob_idx.index(idx)]
    #        #print(pred_future_prob[idx])
    #        #print(color)
    #        #print()
    #    else:
    #        color = (0,0,0,1)
    #    for i in range(len(traj) - 1):
    #        draw_line_with_mask(traj[i], traj[i + 1], color=color, line_width=3)
    
    #draw best 10
    for idx, traj in enumerate(pred_future_traj):
        # calculate color based on probability
        #color = cm.hot(pred_future_prob[idx])
        #print(idx)
        if idx in sorted_prob_idx:
            if sorted_prob_idx.index(idx) <6:
                color = colors[sorted_prob_idx.index(idx)]
            else:
                color = (1,0.93,0.925,1.0)
        else:
            #continue
            color = (1,0.93,0.925,1.0)
        for i in range(len(traj) - 1):
            draw_line_with_mask(traj[i], traj[i + 1], color=color, line_width=3)

    # draw future trajectory
    for idx, traj in enumerate(pred_center_gt_trajs[np.newaxis]):
        draw_trajectory(traj, line_width=2)

    return plt
