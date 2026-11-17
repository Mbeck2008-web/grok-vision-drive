// Non-blocking localhost UDP receiver for scene messages (live mode).
#pragma once
#include <string>

namespace gvd {
class UdpReceiver {
 public:
  ~UdpReceiver();
  // Bind 127.0.0.1:port. port 0 picks an ephemeral port; read it back with port().
  bool open(int port, std::string* err);
  int port() const { return port_; }
  // Drain every queued datagram; keep only the newest. Returns false if none.
  // If drained != nullptr, it receives the number of datagrams consumed.
  bool latest(std::string& out, int* drained = nullptr);
 private:
  long long sock_ = -1;
  int port_ = 0;
};
}  // namespace gvd
