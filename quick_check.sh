#!/bin/bash
echo "======================================"
echo "所有年份文件数量对比"
echo "======================================"
for year in 2016 2017 2018 2019 2020 2021 2022; do
    orig_msg=$(ls /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021/$year/*_message_*.npy 2>/dev/null | wc -l)
    enc_msg=$(ls /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021_encoded/$year/*_message_*.npy 2>/dev/null | wc -l)
    orig_book=$(ls /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021/$year/*_orderbook_*.npy 2>/dev/null | wc -l)
    enc_book=$(ls /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021_encoded/$year/*_orderbook_*.npy 2>/dev/null | wc -l)

    if [ "$orig_msg" -eq "$enc_msg" ] && [ "$orig_book" -eq "$enc_book" ]; then
        status="✓"
    else
        status="✗"
    fi

    echo "$year: msg $orig_msg/$enc_msg, book $orig_book/$enc_book $status"
done
echo "======================================"
echo "总计统计:"
orig_total=$(find /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021 -name "*.npy" | wc -l)
enc_total=$(find /lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021_encoded -name "*.npy" | wc -l)
echo "原始总文件数: $orig_total"
echo "编码总文件数: $enc_total"
if [ "$orig_total" -eq "$enc_total" ]; then
    echo "✓ 文件数量完全匹配！"
else
    echo "✗ 文件数量不匹配，差异: $((orig_total - enc_total))"
fi
echo "======================================"
