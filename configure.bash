# Usage:
# . configure.bash

VDIR="venv"

if [ -d ${VDIR} ]; then
    rm -rf ${DIR}
fi

python -m venv ${VDIR}
if [ $? -ne 0 ]; then  # Since this goes wrong so often!
    echo "Error setting up venv; exit"
    return 1
fi
. ${VDIR}/bin/activate

RQ="requirements-jax.txt"
FIND_LINKS="-f https://storage.googleapis.com/jax-releases/jax_releases.html"

pip install --upgrade pip
pip install --upgrade wheel
pip install -r ${RQ} ${FIND_LINKS}
pip install -e .
pre-commit install
