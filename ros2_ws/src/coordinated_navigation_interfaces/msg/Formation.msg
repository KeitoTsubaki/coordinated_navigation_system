# Formation.msg
#
# Represents the desired configuration summary for a multi-robot formation.

std_msgs/Header header

# Unique identifier for this formation instance.
string formation_id

# Desired yaw angle of the formation relative to the reference frame (radians).
float64 formation_theta

# Optimization cost associated with achieving this formation.
float64 optimization_cost
