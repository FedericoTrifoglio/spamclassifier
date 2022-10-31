import tarfile
import urllib.request
import ssl
import os
import numpy as np

import email
import email.policy

import re
from bs4 import BeautifulSoup
import warnings

def fetch_data(url, extr_path):
    """
    Download compressed tar file (tgz, bz2) and extract its content

    Parameters
    ----------
    url: str
        URL of a compressed tar file
    extr_path: str
        Path of where to extract
    """
    # Choose not to authenticate SSL certificate
    ssl._create_default_https_context = ssl._create_unverified_context
    # Create directory
    if not os.path.isdir(extr_path):
        os.makedirs(extr_path)
    # Define path of local file for download
    compressed_filename = url.split("/")[-1]
    compressed_path = os.path.join(extr_path, compressed_filename)
    # Copy a network object denoted by a URL to a local file
    urllib.request.urlretrieve(url, compressed_path)
    with tarfile.open(compressed_path) as t:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(t, path=extr_path)

def load_email(spam_path, filename):
    """
    Parse a single email from a raw email file

    Parameters
    ----------
    spam_path:
        Path of folder containing filename
    filename: str
        Name of file with email to parse
    
    Returns
    -------
    parsed_email: email.message.EmailMessage object
    """
    with open(os.path.join(spam_path, filename), "rb") as f:
        parsed_email = email.parser.BytesParser(policy=email.policy.default).parse(f)
        return parsed_email

def make_email_example(email, is_spam, clean):
    """
    Make a labelled training/test example for spam classifier from raw emails

    Parameters
    ----------
    email: email.message.EmailMessage object
    is_spam: bool
        Whether is spam or not
    clean: bool
        Whether to clean

    Returns
    -------
    labelled_example: numpy array of shape (1,2)
    """
    # Suppress BeautifulSoup warning when text is just a URL
    warnings.filterwarnings("ignore", category=UserWarning, 
                            module='bs4', 
                            message='[\s\S]*looks like a URL[\s\S]*')
    # Walk through the email message and compile text
    email_text = [" " if email['Subject'] is None else email['Subject']]

    for part in email.walk():
        if isinstance(part.get_payload(), list):
            for subpart in part.get_payload():
                if 'text' in subpart.get_content_type():
                    email_text.append(subpart.get_payload())
        else:
            if 'text' in part.get_content_type():
                email_text.append(part.get_payload())

    email_text = "\n".join(email_text)

    try:
        # parse text
        soup = BeautifulSoup(email_text, 'html.parser')
        email_text = soup.get_text()
    except:
        email_text = email_text

    if clean:
        # remove long sequences of characters (possibly base64 econded images)
        email_text = re.sub(r'\S{20,}(?:\n|$)', '\n', email_text)
        # replace emails and urls with tags
        email_text = re.sub( r'\S*@\S*\s?', ' _EMAIL_ ', email_text)
        email_text = re.sub(r'https?://\S*\s?', ' _URL_ ', email_text)
        # remove sequences of specials
        email_text = re.sub(r'[&#<>{}\[\]\+|\-=_:\\\\]{2,}', ' ', email_text)
        # remove text or code in brackets like [0]
        email_text = re.sub(r'\[[^\[\]]*\]', ' ', email_text)
        # remove > >
        email_text = re.sub(r'(> )+', ' ', email_text)
        # remove multi newlines
        email_text = re.sub(r'\n+', '\n', email_text)
        # remove multi whitespaces
        email_text = " ".join(email_text.split())
        # remove leading and trailing spaces
        email_text = email_text.strip()
        # all lower case
        email_text = email_text.lower()

    labelled_example = np.array([[email_text, 
                                  1 if is_spam else 0]], dtype='object')

    return labelled_example

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, 
                                       cond=None):
    """
    Plot the precisions and recalls functions for different thresholds 
    over the x-axis as well as the point-estimate precision and recall 
    corresponding to a condition
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([0, 1, 0, 1])
    
    if cond is None:
        cond = np.argmax(thresholds >= .5)
        
    thr = thresholds[cond]
    rec = recalls[cond]
    pre = precisions[cond]
    
    plt.plot([thr, thr], [0, pre], "r:")
    plt.plot([0, thr], [pre, pre], "r:")
    plt.plot([thr], [pre], "ro")

    plt.plot([thr, thr], [0, rec], "r:")
    plt.plot([0, thr], [rec, rec], "r:")
    plt.plot([thr], [rec], "ro")

    plt.title(f"At a threshold of {thr:.1f}, precision is {pre:.1%} \
                and recall is {rec:.1%}")
    plt.show()