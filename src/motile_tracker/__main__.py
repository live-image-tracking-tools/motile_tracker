import argparse
import sys

import napari

from napari_track_edit.application_menus.main_app import StartupWidget

# Importing this module registers the Motile tab with napari-track-edit's
# TrackingWidget as a side effect.
import motile_tracker.menus.registration  # noqa: F401


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

    napari.run()


if __name__ == "__main__":
    sys.exit(main())
