// Minimal JSON reader (RFC 8259 subset sufficient for placer configuration files).
#pragma once

#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace dpfpga::json {

class Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value, std::less<>>;

class Value {
 public:
  using Storage = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;

  Value() : v_(nullptr) {}
  explicit Value(Storage v) : v_(std::move(v)) {}

  [[nodiscard]] bool is_null() const { return std::holds_alternative<std::nullptr_t>(v_); }
  [[nodiscard]] bool is_bool() const { return std::holds_alternative<bool>(v_); }
  [[nodiscard]] bool is_number() const { return std::holds_alternative<double>(v_); }
  [[nodiscard]] bool is_string() const { return std::holds_alternative<std::string>(v_); }
  [[nodiscard]] bool is_array() const { return std::holds_alternative<Array>(v_); }
  [[nodiscard]] bool is_object() const { return std::holds_alternative<Object>(v_); }

  [[nodiscard]] double as_number() const {
    if (is_bool()) return std::get<bool>(v_) ? 1.0 : 0.0;
    return std::get<double>(v_);
  }
  [[nodiscard]] int as_int() const { return static_cast<int>(as_number()); }
  [[nodiscard]] bool as_bool() const { return as_number() != 0.0; }
  [[nodiscard]] const std::string& as_string() const { return std::get<std::string>(v_); }
  [[nodiscard]] const Array& as_array() const { return std::get<Array>(v_); }
  [[nodiscard]] const Object& as_object() const { return std::get<Object>(v_); }

 private:
  Storage v_;
};

class Parser {
 public:
  explicit Parser(std::string_view text) : s_(text) {}

  Value parse() {
    Value v = parse_value();
    skip_ws();
    if (pos_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void fail(const char* what) const {
    throw std::runtime_error(std::string("JSON parse error: ") + what + " at offset " + std::to_string(pos_));
  }
  void skip_ws() {
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else if (c == '#') {  // tolerate '#' line comments as a convenience
        while (pos_ < s_.size() && s_[pos_] != '\n') ++pos_;
      } else {
        break;
      }
    }
  }
  char peek() {
    skip_ws();
    if (pos_ >= s_.size()) fail("unexpected end of input");
    return s_[pos_];
  }
  void expect(char c) {
    if (peek() != c) fail("unexpected character");
    ++pos_;
  }
  bool consume_literal(std::string_view lit) {
    if (s_.substr(pos_, lit.size()) == lit) {
      pos_ += lit.size();
      return true;
    }
    return false;
  }

  Value parse_value() {
    switch (peek()) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': return Value(Value::Storage(parse_string()));
      case 't':
        if (consume_literal("true")) return Value(Value::Storage(true));
        fail("bad literal");
      case 'f':
        if (consume_literal("false")) return Value(Value::Storage(false));
        fail("bad literal");
      case 'n':
        if (consume_literal("null")) return Value();
        fail("bad literal");
      default: return parse_number();
    }
  }

  Value parse_number() {
    const char* begin = s_.data() + pos_;
    char* end = nullptr;
    double d = std::strtod(begin, &end);
    if (end == begin) fail("invalid number");
    pos_ += static_cast<size_t>(end - begin);
    return Value(Value::Storage(d));
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (pos_ < s_.size()) {
      char c = s_[pos_++];
      if (c == '"') return out;
      if (c == '\\') {
        if (pos_ >= s_.size()) fail("bad escape");
        char e = s_[pos_++];
        switch (e) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'u': {  // keep BMP code points as UTF-8
            if (pos_ + 4 > s_.size()) fail("bad unicode escape");
            unsigned cp = static_cast<unsigned>(std::stoul(std::string(s_.substr(pos_, 4)), nullptr, 16));
            pos_ += 4;
            if (cp < 0x80) {
              out += static_cast<char>(cp);
            } else if (cp < 0x800) {
              out += static_cast<char>(0xC0 | (cp >> 6));
              out += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
              out += static_cast<char>(0xE0 | (cp >> 12));
              out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
              out += static_cast<char>(0x80 | (cp & 0x3F));
            }
            break;
          }
          default: out += e; break;  // \" \\ \/
        }
      } else {
        out += c;
      }
    }
    fail("unterminated string");
  }

  Value parse_array() {
    expect('[');
    Array arr;
    if (peek() == ']') {
      ++pos_;
      return Value(Value::Storage(std::move(arr)));
    }
    while (true) {
      arr.push_back(parse_value());
      char c = peek();
      ++pos_;
      if (c == ']') break;
      if (c != ',') fail("expected ',' or ']'");
    }
    return Value(Value::Storage(std::move(arr)));
  }

  Value parse_object() {
    expect('{');
    Object obj;
    if (peek() == '}') {
      ++pos_;
      return Value(Value::Storage(std::move(obj)));
    }
    while (true) {
      peek();
      std::string key = parse_string();
      expect(':');
      obj.emplace(std::move(key), parse_value());
      char c = peek();
      ++pos_;
      if (c == '}') break;
      if (c != ',') fail("expected ',' or '}'");
    }
    return Value(Value::Storage(std::move(obj)));
  }

  std::string_view s_;
  size_t pos_ = 0;
};

inline Value parse(std::string_view text) { return Parser(text).parse(); }

}  // namespace dpfpga::json
