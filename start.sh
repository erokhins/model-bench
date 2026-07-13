#!/bin/bash
cd "$(dirname "$0")"
exec python3 load_test.py "$@"