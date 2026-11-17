// Non-blocking localhost UDP receiver for scene messages (live mode).
#pragma once
#include <string>

namespace gvd {
class UdpReceiver {
 public:
  ~UdpReceiver();
  bool open(int port, std::string* err);
  // Drain every queued datagram; keep only the newest. Returns false if none.
  bool latest(std::string& out, int* drained = nullptr);
 private:
  long long sock_ = -1;
};
}  // namespace gvd
