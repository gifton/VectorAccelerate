#!/usr/bin/env perl
use strict; use warnings;
my $current = '';
while (<>) {
  if (/^Test Case '([^']+)' started\./) {
    $current = $1;
    next;
  }
  if (/^Test Case '([^']+)' (passed|failed|skipped)/) {
    $current = '';
    next;
  }
  if ($current ne '' && /((Recall|Accuracy|Precision|F1|AUC|AP|consistency|equivalence)[^\n]*100(?:\.0)?%|100(?:\.0)?%[^\n]*(Recall|Accuracy|Precision|F1|AUC|AP|consistency|equivalence))/i) {
    print "$current\t$_";
  }
}
