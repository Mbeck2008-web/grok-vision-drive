#include "udp.hpp"

#include <vector>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace gvd {

UdpReceiver::~UdpReceiver() {
  if (sock_ < 0) return;
#ifdef _WIN32
  closesocket(static_cast<SOCKET>(sock_));
  WSACleanup();
#else
  close(static_cast<int>(sock_));
#endif
}

bool UdpReceiver::open(int port, std::string* err) {
#ifdef _WIN32
  WSADATA wsa;
  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
    if (err) *err = "WSAStartup failed";
    return false;
  }
  SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (s == INVALID_SOCKET) {
    if (err) *err = "socket failed";
    return false;
  }
  u_long nb = 1;
  ioctlsocket(s, FIONBIO, &nb);
#else
  int s = socket(AF_INET, SOCK_DGRAM, 0);
  if (s < 0) {
    if (err) *err = "socket failed";
    return false;
  }
  fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK);
#endif
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<unsigned short>(port));
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // localhost only
  if (bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
    if (err) *err = "bind 127.0.0.1:" + std::to_string(port) + " failed";
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
    return false;
  }
  sockaddr_in bound{};
#ifdef _WIN32
  int blen = sizeof(bound);
#else
  socklen_t blen = sizeof(bound);
#endif
  if (getsockname(s, reinterpret_cast<sockaddr*>(&bound), &blen) == 0) {
    port_ = static_cast<int>(ntohs(bound.sin_port));
  } else {
    port_ = port;
  }
  sock_ = static_cast<long long>(s);
  return true;
}

bool UdpReceiver::latest(std::string& out, int* drained) {
  if (sock_ < 0) return false;
  static thread_local std::vector<char> buf(65536);
  bool got = false;
  int n_drained = 0;
  for (int i = 0; i < 1024; ++i) {
#ifdef _WIN32
    int n = recv(static_cast<SOCKET>(sock_), buf.data(), static_cast<int>(buf.size()), 0);
#else
    long n = recv(static_cast<int>(sock_), buf.data(), buf.size(), 0);
#endif
    if (n <= 0) break;  // would-block, reset or error: nothing more this frame
    out.assign(buf.data(), static_cast<size_t>(n));
    got = true;
    ++n_drained;
  }
  if (drained) *drained = n_drained;
  return got;
}

}  // namespace gvd
