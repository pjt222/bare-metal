#include <cudnn.h>

#include <cstdio>

int main() {
    std::fprintf(stderr,
                 "cuDNN SDPA reference unavailable: installed cuDNN %ld headers do not expose the graph-based SDPA frontend used by the upstream local-reference sample on this machine.\n",
                 cudnnGetVersion());
    return 0;
}
