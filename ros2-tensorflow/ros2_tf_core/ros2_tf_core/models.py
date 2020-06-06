# Copyright 2020 Alberto Soragna. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from enum import Enum
import os
import sys
import tarfile
import urllib.request
import zipfile

import tensorflow as tf


class SaveLoadFormat(Enum):
    FROZEN_MODEL = 1
    SAVED_MODEL = 2


class ModelDescriptor:

    def __init__(self):
        self.url = None
        self.download_directory = None
        self.model_filename = None
        self.model_path = None
        self.save_load_format = None
        self.label_path = None
        self.description = ''

    def from_url(self, url, label_path,
                 download_directory='.', model_filename='',
                 save_load_format=SaveLoadFormat.FROZEN_MODEL, description=''):
        if self.model_path is not None:
            raise NameError('Calling from url when a path has been already provided')
        self.url = url
        self.label_path = label_path
        self.download_directory = download_directory
        self.model_filename = model_filename
        self.save_load_format = save_load_format
        self.description = description
        return self

    def from_path(self, model_path, label_path,
                  save_load_format=SaveLoadFormat.FROZEN_MODEL, description=''):
        if self.url is not None:
            raise NameError('Calling from path when a url has been already provided')
        self.model_path = model_path
        self.label_path = label_path
        self.save_load_format = save_load_format
        self.description = description
        return self

    def compute_model_path(self):
        # If the model path is not specified, get it from url
        if self.model_path is None:
            if self.url is None:
                raise NameError('No url or file path have been provided')
            downloaded_path = maybe_download_and_extract(self.url, self.download_directory)

            if os.path.isdir(downloaded_path):
                joined_path = os.path.join(downloaded_path, self.model_filename)
                if not os.path.exists(joined_path):
                    print(f'Model filename: {self.model_filename}')
                    print(f'Model path: {joined_path}')
                    raise ValueError(
                        'The provided model filename does not exists in the downloaded directory')
                self.model_path = joined_path
            else:
                if self.model_filename and self.model_filename != downloaded_path:
                    print(f'Model filename: {self.model_filename}')
                    print(f'Downloaded path: {downloaded_path}')
                    raise ValueError(
                        'The provided model filename has not been found')
                self.model_path = downloaded_path

        return self.model_path

    def compute_label_path(self):
        return self.label_path


def maybe_download_and_extract(url_source, download_directory,
                               filename=None, extract=True, expected_bytes=None):
    """
    Check if file exists in download_directory otherwise tries to dowload and extract it.

    Parameters
    ----------
    url_source : str
        The URL to download the file from
    download_directory : str
        A folder path to search for the file in and dowload the file to
    filename : str
        The name of the (to be) dowloaded file.
    extract : boolean
        If True, tries to uncompress the dowloaded file, default is True.
        Supported formats are ".tar.gz/.tar.bz2" or ".zip" files.
        If different format, extraction is skipped.
    expected_bytes : int or None
        If set tries to verify that the downloaded file is of the specified size,
        otherwise raises an Exception, defaults is None which corresponds to no check.

    Returns
    -------
    str
        File path of the dowloaded (uncompressed) file.

    Examples
    --------
    >>> res = maybe_download_and_extract(
    ...     url_source='http://yann.lecun.com/exdb/mnist/',
    ...     download_directory='data/')

    """
    # Create a download directory if not already existing
    if not os.path.exists(download_directory):
        os.makedirs(download_directory)

    if filename is None:
        # Get the filename from the URL
        filename_with_extension = url_source.split('/')[-1]
        filename = filename_with_extension.split('.')[0]
    else:
        filename_with_extension = filename

    filepath = os.path.join(download_directory, filename)
    # Download the file if not already existing
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            percentage = float(count * block_size) / float(total_size) * 100.0
            sys.stdout.write(f'\r>> Downloading {filename} {percentage:.1f}%')
            sys.stdout.flush()

        # The downloaded file must end with a correct extension to avoid problems
        filepath_with_extension = os.path.join(download_directory, filename_with_extension)
        urllib.request.urlretrieve(url_source, filepath_with_extension, _progress)

        statinfo = os.stat(filepath_with_extension)
        print(f'Succesfully downloaded {filename} {statinfo.st_size} bytes.')
        if (not (expected_bytes is None) and (expected_bytes != statinfo.st_size)):
            raise Exception(f'Failed to verify {filename}. Can you get to it with a browser?')
        if (extract):
            try:
                print(f'Trying to extract archive {filepath_with_extension}')
                if tarfile.is_tarfile(filepath_with_extension):
                    with tarfile.open(filepath_with_extension, 'r') as tf:
                        # If the archive contains more than 1 file, extract into a directory
                        if archive_has_toplevel_dir(tf.getnames()):
                            tf.extractall(download_directory)
                        else:
                            tf.extractall(os.path.join(download_directory, filename))
                    # Remove the downloaded file if we extracted it
                    os.remove(filepath_with_extension)
                elif zipfile.is_zipfile(filepath_with_extension):
                    with zipfile.ZipFile(filepath_with_extension) as zf:
                        # If the archive contains more than 1 file, extract into a directory
                        if archive_has_toplevel_dir(zf.namelist()):
                            zf.extractall(download_directory)
                        else:
                            zf.extractall(os.path.join(download_directory, filename))
                    # Remove the downloaded file if we extracted it
                    os.remove(filepath_with_extension)
                else:
                    print('Skipping file extraction as format not supported')
                    # The file has not been extracted, so we keep the extension
                    filepath = filepath_with_extension
            except Exception as e:
                # Remove the downloaded file before raising the exception
                os.remove(filepath_with_extension)
                raise(e)

    return filepath


def archive_has_toplevel_dir(names):
    # If there is only 1 file, we assume it is not a directory
    if len(names) == 1:
        return False

    # If there is not a common path among all the files in the archive,
    # then there is not a top level directory
    common_path = os.path.commonpath(names)
    if not common_path or common_path == '.':
        return False

    # If we are here, then the archive has a toplevel directory
    return True


def load_frozen_model(frozen_model_path):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(frozen_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    session = tf.compat.v1.Session(graph=graph)
    return graph, session


def load_saved_model(saved_model_dir):

    model = tf.saved_model.load(str(saved_model_dir))
    model = model.signatures['serving_default']

    return model
