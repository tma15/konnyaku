# -*- coding: utf-8 -*-
from setuptools import setup

setup(name='konnyaku',
    packages=[
        'konnyaku',
        'konnyaku.encoders',
        'konnyaku.decoders',
    ],
    entry_points = {
        'console_scripts': [
            'konnyaku_train=konnyaku.__train__:main',
            'konnyaku_eval=konnyaku.__eval__:main',
        ],
    },
    zip_safe=False,
)
