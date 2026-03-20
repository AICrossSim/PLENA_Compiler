awk '
/^; unknown nodes:/ {inblk=1; next}

inblk && /^[[:space:]]*$/ { next }

inblk && /^;[[:space:]]{3,}/ {
  if (match($0, /\([^()]*\)[[:space:]]*$/)) {
    t = substr($0, RSTART+1, RLENGTH-2)
    gsub(/[[:space:]]+$/, "", t)
    if (!(t in seen)) { seen[t]=1; order[++n]=t }
  }
  next
}

inblk && (/^; --- \[/ || /^; \[ERROR/ || /^; =====/ ) { inblk=0 }

inblk { inblk=0 }

END { for (i=1; i<=n; i++) print order[i] }
' ../vlm_output.asm > ./unknowns.asm