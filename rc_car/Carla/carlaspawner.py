import carla
import random
import time
import numpy as np
import cv2

# ---------------- 사용자 입력 ----------------
n = int(input("자율주행 차량 수 (n): "))
m = int(input("도보 시민 수 (m): "))
fog = float(input("안개 농도 (0.0 ~ 1.0): "))
rain = float(input("강수량 (0.0 ~ 1.0): "))
sun = float(input("태양 고도 (-90.0 : 밤 ~ 90.0 : 낮): "))

# ---------------- CARLA 연결 및 맵 로딩 ----------------
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.load_world("Town03")
blueprint_library = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
random.shuffle(spawn_points)

# ---------------- 날씨 설정 ----------------
weather = world.get_weather()
weather.fog_density = fog * 100.0
weather.fog_distance = 5.0
weather.fog_falloff = 0.3
weather.precipitation = rain * 100
weather.precipitation_deposits = rain * 100
weather.wetness = rain * 100
weather.cloudiness = 100.0
weather.wind_intensity = 60.0
weather.sun_altitude_angle = sun
world.set_weather(weather)
print("[INFO] 날씨 설정 완료")

# ---------------- 유틸: 랜덤 위치 쌍 ----------------
def get_random_location_pair(min_distance=30.0):
    a = world.get_random_location_from_navigation()
    b = world.get_random_location_from_navigation()
    while b.distance(a) < min_distance:
        b = world.get_random_location_from_navigation()
    return a, b

# ---------------- 차량 필터링 (일반 승용차만 허용) ----------------
allowed_vehicle_ids = [
    'vehicle.audi.a2',
    'vehicle.bmw.grandtourer',
    'vehicle.chevrolet.impala',
    'vehicle.citroen.c3',
    'vehicle.dodge.charger_2020',
    'vehicle.genesis.g80',
    'vehicle.jeep.wrangler_rubicon',
    'vehicle.lincoln.mkz_2020',
    'vehicle.mercedes.coupe',
    'vehicle.mini.cooper_s',
    'vehicle.ford.mustang',
    'vehicle.nissan.micra',
    'vehicle.peugeot.208',
    'vehicle.seat.leon',
    'vehicle.tesla.model3',
    'vehicle.toyota.prius',
    'vehicle.volkswagen.passat',
    'vehicle.volkswagen.t2',
    'vehicle.volkswagen.tiguan'
]

vehicle_bps = [bp for bp in blueprint_library.filter('vehicle.*') if bp.id in allowed_vehicle_ids]
vehicles = []
vehicle_routes = []

for i in range(n):
    bp = random.choice(vehicle_bps)
    transform = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(bp, transform)
    if vehicle:
        vehicle.set_autopilot(True)
        a, b = get_random_location_pair()
        vehicle_routes.append((vehicle, [a, b]))
        vehicles.append(vehicle)
print(f"[INFO] 차량 {len(vehicles)}대 생성 완료")

# ---------------- 시민 스폰 ----------------
walker_bps = blueprint_library.filter('walker.pedestrian.*')
controller_bp = blueprint_library.find('controller.ai.walker')
walkers = []
walker_controllers = []
walker_routes = []

for i in range(m):
    a, b = get_random_location_pair()
    bp = random.choice(walker_bps)
    walker = world.try_spawn_actor(bp, carla.Transform(a))
    if walker:
        controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        controller.start()
        controller.go_to_location(b)
        controller.set_max_speed(1.0 + random.random())
        walkers.append(walker)
        walker_controllers.append(controller)
        walker_routes.append((controller, [a, b]))
print(f"[INFO] 시민 {len(walkers)}명 생성 완료")

# ---------------- 내 차량 + 카메라 센서 ----------------
my_bp = blueprint_library.find('vehicle.tesla.model3')
my_transform = random.choice(spawn_points)
my_vehicle = world.try_spawn_actor(my_bp, my_transform)
if my_vehicle:
    my_vehicle.set_autopilot(True)

camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x", "500")
camera_bp.set_attribute("image_size_y", "500")
camera_bp.set_attribute("fov", "90")
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=my_vehicle)

# ---------------- 카메라 콜백 ----------------
def process_camera(image):
    try:
        img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
        cv2.imshow("Camera View", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

        # ============================
        # 여기에 서버 전송 코드 작성
        # 예: send_to_server(img)
        # ============================

    except Exception as e:
        print(f"[ERROR] 카메라 예외: {e}")

camera.listen(lambda image: process_camera(image))
print("[INFO] 내 차량 및 카메라 설정 완료")

# ---------------- 시민 루프 ----------------
try:
    walker_targets = {id(c): 0 for c, _ in walker_routes}
    print("[INFO] 시뮬레이터 실행 중... Ctrl+C로 종료")

    while True:
        time.sleep(1.0)

        for controller, (a, b) in walker_routes:
            walker = controller._parent
            current_target = walker_targets[id(controller)]
            target = a if current_target == 0 else b
            if walker.get_location().distance(target) < 2.0:
                next_target = 1 - current_target
                controller.go_to_location(b if next_target == 1 else a)
                walker_targets[id(controller)] = next_target

except KeyboardInterrupt:
    print("\n[INFO] 종료 요청 감지. 리소스 정리 중...")

finally:
    camera.stop()
    for ctrl in walker_controllers:
        ctrl.stop()
    actors = vehicles + walkers + walker_controllers + [my_vehicle, camera]
    for a in actors:
        if a:
            a.destroy()
    cv2.destroyAllWindows()
    print("[INFO] 모든 리소스 정리 완료")
