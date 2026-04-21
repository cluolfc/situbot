from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['situbot', 'situbot.perception', 'situbot.reasoning',
              'situbot.planning', 'situbot.execution', 'situbot.evaluation',
              'situbot.utils'],
    package_dir={'': 'src'}
)

setup(**d)
