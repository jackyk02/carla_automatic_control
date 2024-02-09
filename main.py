from automatic_control import Args, logging, torch, game_start, game_step


def main():
    """Main method"""
    args = Args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    logging.info("listening to server %s:%s", args.host, args.port)
    yolo_model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s", pretrained=True)

    try:
        clock, world, controller, display, agent, spawn_points = game_start(
            args)
        while True:
            # tick
            game_step(clock, world, controller, display, agent, spawn_points)

            # control
            control = agent.run_step()
            control.manual_gear_shift = False
            world.player.apply_control(control)

            # yolo output
            if world.camera_manager.img is not None:
                results = yolo_model(world.camera_manager.img)
                results.print()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
