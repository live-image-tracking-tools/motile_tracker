import argparse
import sys

import napari

from napari_track_edit.application_menus.main_app import StartupWidget

from motile_tracker.menus.motile_widget import MotileWidget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "tracking", "editing"],
        default="all",
    )

    args, _ = parser.parse_known_args()

    viewer = napari.Viewer()
    StartupWidget(viewer, mode=args.mode)
    viewer.window.add_dock_widget(MotileWidget(viewer), name="Tracking")

    napari.run()


if __name__ == "__main__":
    sys.exit(main())
