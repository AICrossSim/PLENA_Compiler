awk '
/^; unknown nodes:/ {inblk=1; next}

# unknown 区块内：忽略空行
inblk && /^[[:space:]]*$/ { next }

# unknown 区块内：条目行 ";   name (Type)"
inblk && /^;[[:space:]]{3,}/ {
  if (match($0, /\([^()]*\)[[:space:]]*$/)) {
    t = substr($0, RSTART+1, RLENGTH-2)   # 拿到括号里的 Type
    gsub(/[[:space:]]+$/, "", t)
    if (!(t in seen)) { seen[t]=1; order[++n]=t }
  }
  next
}

# unknown 区块结束条件：遇到明显的新 section（你也可以按需要加更多模式）
inblk && (/^; --- \[/ || /^; \[ERROR/ || /^; =====/ ) { inblk=0 }

# 或者遇到非空、且不是条目行的注释/内容，也结束
inblk { inblk=0 }

END { for (i=1; i<=n; i++) print order[i] }
' ../vlm_output.asm > ./unknowns.asm