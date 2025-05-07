from pkg_resources import parse_version
from configparser import ConfigParser
import setuptools, shlex
assert parse_version(setuptools.__version__)>=parse_version('36.2')

# note: all settings are in settings.ini; edit there, not here
config = ConfigParser(delimiters=['='])
config.read('settings.ini', encoding='utf-8')
cfg = config['DEFAULT']

cfg_keys = 'version description keywords author author_email'.split()
expected = cfg_keys + "lib_name user branch license status min_python audience language".split()
for o in expected: assert o in cfg, "missing expected setting: {}".format(o)
setup_cfg = {o:cfg[o] for o in cfg_keys}

licenses = {
    'apache2': ('Apache Software License 2.0','OSI Approved :: Apache Software License'),
    'mit': ('MIT License', 'OSI Approved :: MIT License'),
    'gpl2': ('GNU General Public License v2', 'OSI Approved :: GNU General Public License v2 (GPLv2)'),
    'gpl3': ('GNU General Public License v3', 'OSI Approved :: GNU General Public License v3 (GPLv3)'),
    'bsd3': ('BSD License', 'OSI Approved :: BSD License'),
}
statuses = [ '1 - Planning', '2 - Pre-Alpha', '3 - Alpha',
    '4 - Beta', '5 - Production/Stable', '6 - Mature', '7 - Inactive' ]
py_versions = '3.6 3.7 3.8 3.9 3.10 3.11 3.12'.split()

requirements = shlex.split(cfg.get('requirements', ''))
if cfg.get('pip_requirements'): requirements += shlex.split(cfg.get('pip_requirements', ''))
min_python = cfg['min_python']
lic = licenses.get(cfg['license'].lower(), (cfg['license'], None))
dev_requirements = (cfg.get('dev_requirements') or '').split()

package_data = dict()
pkg_data = cfg.get('package_data', None)
if pkg_data:
    package_data[cfg['lib_name']] =  pkg_data.split() # split as multiple files might be listed
# Add package data to setup_cfg for setuptools.setup(..., **setup_cfg)
setup_cfg['package_data'] = package_data

setuptools.setup(
    name = cfg['lib_name'],
    license = lic[0],
    classifiers = [
        'Development Status :: ' + statuses[int(cfg['status'])],
        'Intended Audience :: ' + cfg['audience'].title(),
        'Natural Language :: ' + cfg['language'].title(),
    ] + ['Programming Language :: Python :: '+o for o in py_versions[py_versions.index(min_python):]] + (['License :: ' + lic[1] ] if lic[1] else []),
    url = cfg['git_url'],
    packages = setuptools.find_packages(),
    include_package_data = True,
    install_requires = requirements,
    extras_require={ 'dev': dev_requirements },
    dependency_links = cfg.get('dep_links','').split(),
    python_requires  = '>=' + cfg['min_python'],
    long_description = open('README.md', encoding='utf-8').read(),
    long_description_content_type = 'text/markdown',
    zip_safe = False,
    entry_points = {
        'console_scripts': cfg.get('console_scripts','').split(),
        'nbdev': [f'{cfg.get("lib_path")}={cfg.get("lib_path")}._modidx:d']
    },
    **setup_cfg)


