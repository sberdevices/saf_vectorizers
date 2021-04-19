from setuptools import find_packages, setup


setup(
    name="saf_vectorizers",
    author="SberDevices",
    author_email="developer@sberdevices.ru",
    description="SAF Vectorizers - это плагин для SmartApp Framework, осуществляющий векторизацию "
                "(получение embedding'ов) текстов с помощью различных моделей",
    long_description_content_type="text/markdown",
    license="sberpl-2",
    packages=find_packages(exclude=[]),
    include_package_data=True,
    install_requires=[
        "smart_app_framework",
        "gdown==3.12.2",
        "gensim==4.0.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
    ]
)
