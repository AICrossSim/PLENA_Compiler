awk '
function flush() {
  if (block != "") { print block; block="" }
}

/^; --- \[/ {
  flush()
  inblock=1
  block = $0 "\n"
  next
}

inblock && /^;     / {
  block = block $0 "\n"
  next
}

inblock && /^; \[ERROR/ {
  block = block $0 "\n"
  next
}

inblock {
  flush()
  inblock=0
  next
}

END { flush() }
' ../vlm_output.asm > ./sampled_nodes.asm