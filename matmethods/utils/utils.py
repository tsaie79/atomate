# coding: utf-8

from __future__ import division, print_function, unicode_literals, \
    absolute_import

import six
import logging
import sys
import os
import glob
import shutil
import paramiko

__author__ = 'Anubhav Jain, Kiran Mathew'
__email__ = 'ajain@lbl.gov, kmathew@lbl.gov'


def env_chk(val, fw_spec, strict=True):
    """
    env_chk() is a way to set different values for a property depending
    on the worker machine. For example, you might have slightly different
    executable names or scratch directories on different machines.

    env_chk() works using the principles of the FWorker env in FireWorks.
    For more details, see:
    https://pythonhosted.org/FireWorks/worker_tutorial.html

    This helper method translates string values that look like this:
    ">>ENV_KEY<<"
    to the contents of:
    fw_spec["_fw_env"][ENV_KEY]

    Since the latter can be set differently for each FireWorker, one can
    use this method to translate a single value into multiple possibilities,
    thus achieving different behavior on different machines.

    Args:
        val: any value, with ">><<" notation reserved for special env lookup
            values
        fw_spec: fw_spec where one can find the _fw_env keys
        strict(bool): if True, errors if env value cannot be found
    """

    if isinstance(val, six.string_types) and val.startswith(
            ">>") and val.endswith("<<"):
        if strict:
            return fw_spec['_fw_env'][val[2:-2]]
        return fw_spec.get('_fw_env', {}).get(val[2:-2])
    return val


def get_logger(name, level=logging.DEBUG,
               format='%(asctime)s %(levelname)s %(name)s %(message)s',
               stream=sys.stdout):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(format)
    sh = logging.StreamHandler(stream=stream)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


class MMos(object):
    """
    paramiko wrapper
    """

    def __init__(self, filesystem=None, pkey_file="~/.ssh/id_rsa"):
        """
        filesystem (string): remote filesystem, e.g. username@remote_host
        pkey_file (string): path to the private key file.
            Note: passwordless ssh login must be setup
        """
        self.ssh = None
        username = None
        host = None
        if filesystem:
            tokens = filesystem.split('@')
            username = tokens[0]
            host = tokens[1]
        if username and host:
            self.ssh = self._get_ssh_connection(username, host, pkey_file)

    def _get_ssh_connection(self, username, host, pkey_file):
        """
        Setup ssh connection using paramiko and return the channel
        """
        privatekeyfile = os.path.expanduser(pkey_file)
        if not os.path.exists(privatekeyfile):
            possible_keys = ["~/.ssh/id_rsa", "~/.ssh/id_dsa", "/etc/ssh/id_rsa", "/etc/ssh/id_dsa"]
            for key in possible_keys:
                if os.path.exists(os.path.expanduser(key)):
                    privatekeyfile = os.path.expanduser(key)
                    break
        tokens = privatekeyfile.split("id_")
        try:
            if tokens[1] == "rsa":
                mykey = paramiko.RSAKey.from_private_key_file(privatekeyfile)
            elif tokens[1] == "dsa":
                mykey = paramiko.DSSKey.from_private_key_file(privatekeyfile)
            else:
                print("Unknown private key format. Must be either rsa(preferred) or dsa")
        except:
            print("Found the private key file {}, but not able to load".format(pkey_file))
            return None
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(host, username=username, pkey=mykey)
        except paramiko.SSHException:
            print("Connection Error: host: {}, username: {}".format(host, username))
            return None
        return ssh

    def listdir(self, ldir):
        """
        Wrapper of getting the directory listing from either the local or
        remote filesystem.

        Args:
            ldir (string): full path to the directory

        Returns:
            list of filenames
        """
        if self.ssh:
            try:
                #command = ". ./.bashrc; for i in {}/*; do readlink -f $i; done".format(ldir)
                command = ". ./.bashrc; ls {}".format(ldir)
                stdin, stdout, stderr = self.ssh.exec_command(command)
                return [l.split('\n')[0] for l in stdout]
            except:
                print("paramiko connection error. Make sure that passwordless ssh login is setup and "
                      "yor private key is in standard location. e.g. '~/.ssh/id_rsa'")
                return []
        else:
            return [f for f in os.listdir(ldir)]


    def copy(self, source, dest):
        """
        Wrapper for copying from source to destination. The source can be
        a remote filesystem

        Args:
            source (string): source full path
            dest (string): destination file full path

        """
        if self.ssh:
            try:
                command = ". ./.bashrc; readlink -f {}".format(source)
                stdin, stdout, stderr = self.ssh.exec_command(command)
                source_full_path = [l.split('\n')[0] for l in stdout]
                sftp = self.ssh.open_sftp()
                sftp.get(source_full_path[0], dest)
            except:
                print("paramiko connection error. Make sure that passwordless ssh login is setup and "
                      "yor private key is in standard location. e.g. '~/.ssh/id_rsa'")
                raise IOError
        else:
            shutil.copy2(source, dest)


    def abspath(self, path):
        """
        return the absolute path
        """
        if self.ssh:
            try:
                command = ". ./.bashrc; readlink -f {}".format(path)
                stdin, stdout, stderr = self.ssh.exec_command(command)
                full_path = [l.split('\n')[0] for l in stdout]
                return full_path[0]
            except:
                print("paramiko connection error. Make sure that passwordless ssh login is setup and "
                      "yor private key is in standard location. e.g. '~/.ssh/id_rsa'")
                raise IOError
        else:
            return os.path.abspath(path)


    def glob(self, path):
        """
        return the absolute path
        """
        if self.ssh:
            try:
                command = ". ./.bashrc; for i in $(ls {}); do readlink -f $i; done".format(path)
                stdin, stdout, stderr = self.ssh.exec_command(command)
                return [l.split('\n')[0] for l in stdout]
            except:
                print("paramiko connection error. Make sure that passwordless ssh login is setup and "
                      "yor private key is in standard location. e.g. '~/.ssh/id_rsa'")
                raise IOError
        else:
            return glob.glob(path)

