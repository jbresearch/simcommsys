#!/bin/bash

git filter-repo --commit-callback '
from datetime import datetime

if b"Mark" in commit.author_name:
    commit.author_name = b"Mark Mizzi"
    commit.committer_name = b"Mark Mizzi"

    commit_date = datetime.fromtimestamp(int(commit.author_date.decode().split()[0]))
    print(commit_date)
    if commit_date < datetime(2024, 4, 29):
        commit.author_email = b"mark.mizzi.19@um.edu.mt"
        commit.committer_email = b"mark.mizzi.19@um.edu.mt"
    else:
        commit.author_email = b"mark.mizzi@um.edu.mt"
        commit.committer_email = b"mark.mizzi@um.edu.mt"
' --force
