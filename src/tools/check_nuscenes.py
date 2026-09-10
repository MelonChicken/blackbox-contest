from nuscenes.nuscenes import NuScenes

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=r"..\..\data\raw\stage3\nuscenes",
    verbose=True,
)

print("Scenes:", len(nusc.scene))
print("Samples:", len(nusc.sample))

scene = nusc.scene[0]
print(scene["name"])
print(scene["description"])

scene = nusc.scene[0]

sample = nusc.get("sample", scene["first_sample_token"])
cam_token = sample["data"]["CAM_FRONT"]

frames = []

while cam_token:
    sd = nusc.get("sample_data", cam_token)

    frames.append({
        "timestamp": sd["timestamp"],
        "filename": sd["filename"],
        "is_key_frame": sd["is_key_frame"],
    })

    cam_token = sd["next"]

print("CAM_FRONT frames:", len(frames))
print("Keyframes:", sum(x["is_key_frame"] for x in frames))
print("Non-keyframes:", sum(not x["is_key_frame"] for x in frames))

for x in frames[:10]:
    print(x)
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

nusc_can = NuScenesCanBus(
    dataroot=r"..\..\data\raw\stage3\nuscenes"
)

scene_name = scene["name"]

pose = nusc_can.get_messages(
    scene_name,
    "pose",
)

steer = nusc_can.get_messages(
    scene_name,
    "steeranglefeedback",
)

print("scene:", scene_name)
print("pose count:", len(pose))
print("steer count:", len(steer))

print("\n=== pose ===")
print(pose[0])

print("\n=== steer ===")
print(steer[0])
cam_start = frames[0]["timestamp"]
cam_end = frames[-1]["timestamp"]

pose_start = pose[0]["utime"]
pose_end = pose[-1]["utime"]

steer_start = steer[0]["utime"]
steer_end = steer[-1]["utime"]

print("CAM   :", cam_start, cam_end)
print("POSE  :", pose_start, pose_end)
print("STEER :", steer_start, steer_end)