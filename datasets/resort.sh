dir="files"
otp="outputs"
target="mini-test.jsonl"

awk 'NF > 0 {print}' $dir/$target $dir/$target $dir/$target | head -c -1 > $otp/repeated-$target
# 将 repeated-data.jsonl 文件的所有行随机打乱，并将结果输出到 shuffled-data.jsonl 文件中
sort -R $otp/repeated-$target | head -c -1 > $otp/shuffled-$target