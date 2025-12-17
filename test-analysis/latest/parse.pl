#!/usr/bin/env perl
use strict; use warnings;
while (<>) { if (/^Test Case '([^']+)' (passed|failed|skipped) \(([0-9.]+) seconds\)\./) { print "$2\t$1\t$3\n"; } }
