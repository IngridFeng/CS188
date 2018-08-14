#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CS 188 Local Submission Autograder
Written by the CS 188 Staff

==============================================================================
   _____ _              _ 
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |      
                 |_|      

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64
exec(bz2.decompress(base64.b64decode('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')))

