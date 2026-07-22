from glob import glob

from setuptools import find_packages, setup

package_name = "perception_camera"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Blacknode",
    maintainer_email="temiroff@gmail.com",
    description="Generic USB camera publisher for Blacknode Vision workflows.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "usb_camera = perception_camera.usb_camera:main",
        ],
    },
)
