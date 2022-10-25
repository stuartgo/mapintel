import os
import nox

os.environ.update({'PDM_IGNORE_SAVED_PYTHON': '1'})

@nox.session(python=['3.9', '3.10'])
def check_quality(session):
    session.run('pdm', 'install', '-G', 'tests', '-G', 'duty', external=True)
    session.run('pdm', 'run', 'duty', 'check-quality', external=True)


@nox.session(python=['3.9', '3.10'])
def check_docs(session):
    session.run('pdm', 'install', '-G', 'tests', '-G', 'duty', external=True)
    session.run('pdm', 'run', 'duty', 'check-docs', external=True)


@nox.session(python=['3.9', '3.10'])
def check_types(session):
    session.run('pdm', 'install', '-G', 'tests', '-G', 'duty', external=True)
    session.run('pdm', 'run', 'duty', 'check-types', external=True)


@nox.session(python=['3.9', '3.10'])
def test(session):
    session.run('pdm', 'install', '-G', 'tests', '-G', 'duty', external=True)
    session.run('pdm', 'run', 'duty', 'test', external=True)


@nox.session(python=['3.9', '3.10'])
def check(session):
    session.run('pdm', 'install', '-G', 'tests', '-G', 'duty', external=True)
    session.run('pdm', 'run', 'duty', 'check-quality', 'check-types', 'check-docs', external=True)
