# coordinated_navigation_interfaces

This package provides ROS 2 interface definitions for coordinated multi-robot navigation. The `Formation` message captures the desired configuration metadata for robots within a formation.

## `Formation` message

| Field | Type | Description |
| ----- | ---- | ----------- |
| `header` | `std_msgs/Header` | Timestamp and frame reference for the formation specification. |
| `formation_id` | `string` | Unique identifier for this formation instance. |
| `formation_theta` | `float64` | Desired yaw angle of the formation relative to the reference frame (radians). |
| `optimization_cost` | `float64` | Optimization cost or objective value associated with realizing the formation. |
