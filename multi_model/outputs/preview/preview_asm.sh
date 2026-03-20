in="../vlm_output.asm"; out="./sampled.asm"; n=100; \
lc=$(wc -l < "$in"); mid=$((lc/2)); \
{ echo "; ===== BEGIN: first ${n} lines ====="; head -n "$n" "$in"; \
  echo -e "\n; ===== MIDDLE: around line ${mid} (±${n}/2) ====="; \
  sed -n "$((mid-n/2)),$((mid+n/2-1))p" "$in"; \
  echo -e "\n; ===== END: last ${n} lines ====="; tail -n "$n" "$in"; \
} > "$out"