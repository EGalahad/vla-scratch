"""Common key names used by Libero datasets."""

# Camera stream names surfaced to downstream consumers.
CAM_FRONT_KEY = "cam_front"
CAM_WRIST_KEY = "cam_arm"
TASK_NAME_KEY = "task"

# State
ARM_STATE_CART_POS_KEY = "arm_state_cart_pos"
ARM_STATE_CART_ROT_KEY = "arm_state_cart_rot"
ARM_STATE_JOINT_POS_KEY = "arm_state_joint_pos"  # joints not present in dataset
GRIPPER_STATE_QPOS_KEY = "gripper_state_qpos"


# Commands
ARM_CMD_CART_POS_KEY = "arm_cmd_cart_pos"
ARM_CMD_CART_ROT_KEY = "arm_cmd_cart_rot"
ARM_CMD_JOINT_POS_KEY = "arm_cmd_joint_pos"  # joints not present in dataset
GRIPPER_CMD_ACTION_KEY = "gripper_cmd_action"
