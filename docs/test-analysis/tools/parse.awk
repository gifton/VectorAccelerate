BEGIN { FS="'" }
/^Test Case / && /(passed|failed|skipped)/ {
  name=$2;
  status="";
  if ($0 ~ /passed/) status="passed";
  else if ($0 ~ /failed/) status="failed";
  else if ($0 ~ /skipped/) status="skipped";
  dur="";
  if (match($0, /\(([0-9.]+) seconds\)\./, t)) { dur=t[1]; }
  if (name != "" && dur != "") {
    print dur "\t" status "\t" name;
  }
}
