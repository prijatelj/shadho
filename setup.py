#!/usr/bin/env python
from setuptools import setup
from setuptools.command.install import install

try:
    import configparser
except:
    import ConfigParser as configparser
import os
import site
import shutil
import subprocess
import sys


MAJ = sys.version_info[0]
SHADHO_DIR = os.path.join(os.environ['HOME'], '.shadho')
DEFAULT_CONFIG = {
    'global': {
        'wrapper': 'shadho_run_task.py',
        'output': 'out.tar.gz',
        'resultfile': 'performance.json',
        'minval': 'loss'
    },
    'workqueue': {
        'port': '9123',
        'name': 'shadho_master',
        'exclusive': 'yes',
        'shutdown': 'yes',
        'catalog': 'no',
        'logfile': 'shadho_master.log',
        'debugfile': 'shadho_master.debug',
        'password': 'no'
    },
    'backend': {
        'type': 'json'
    }
}


class InstallCCToolsCommand(install):
    """Helper to install CCTools.
    """
    description = "Install CCTools and set up SHADHO working directory"
    def run(self):
        """Install WorkQueue from the CCTools suite to site-packages.

        SHADHO uses WorkQueue, a member of the `CCTools software suite<http://ccl.cse.nd.edu>`,
        to manage the distributed computing environment. This function installs
        CCTools and moves the related Python module and shared library to the
        site-packages directory.
        """
        print('Installing CCTools suite')
        global MAJ
        global SHADHO_DIR
        global DEFAULT_CONFIG
        # CCTools distinguishes between Python 2/3 SWIG bindings, and the
        # Python 3 bindings require extra effort. Install based on the user's
        # version
        cfg = configparser.ConfigParser()
        if MAJ == 3:
            try:
                import work_queue
                print("Found Work Queue, skipping install")
            except ImportError:
                subprocess.call(['bash', 'install_cctools.sh', 'py3'])
            cfg.read_dict(DEFAULT_CONFIG)
        else:
            try:
                import work_queue
                print("Found Work Queue, skipping install")
            except ImportError:
                subprocess.call(['bash', 'install_cctools.sh'])
            for key, val in DEFAULT_CONFIG.iteritems():
                cfg.add_section(key)
                for k, v in val.iteritems():
                    cfg.set(key, k, v)
        print('Installing shadho_worker')
        shutil.copy(os.path.join('.', 'scripts', 'shadho_run_task.py'),
                    SHADHO_DIR)

        print('Copying default .shadhorc to home directory')
        home = os.path.expanduser(os.environ['HOME'] if 'HOME' in os.environ
                                      else os.environ['USERPROFILE'])
        with open(os.path.join(home, '.shadhorc'), 'w') as f:
            cfg.write(f)
        install.run(self)


setup(
    name='shadho',
    version='0.1a1',
    description='Hyperparameter optimizer with distributed hardware at heart',
    url='https://github.com/jeffkinnison/shadho',
    author='Jeff Kinnison',
    author_email='jkinniso@nd.edu',
    packages=['shadho'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Users',
        'License :: MIT',
        'Topic :: Machine Learning :: Hyperparameter Optimization',
        'Topic :: Distributed Systems :: Task Allocation',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    keywords='machine_learning hyperparameters distributed_computing',
    install_requires=[
        'scipy>=0.18.1',
        'numpy>=1.12.0',
        'scikit-learn>=0.18.1',
        'pandas>=0.18.1'
    ],
    extras_require={
        'test': ['nose', 'coverage']
    },
    cmdclass={'install': InstallCCToolsCommand}
)
