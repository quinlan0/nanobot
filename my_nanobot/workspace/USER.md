# User Profile

Information about the user to help personalize interactions.

## Basic Information

- **Name**: (your name)
- **Timezone**: (your timezone, e.g., UTC+8)
- **Language**: (preferred language)

## Preferences

### Communication Style

- [ ] Casual
- [ ] Professional
- [ ] Technical

### Response Length

- [ ] Brief and concise
- [ ] Detailed explanations
- [ ] Adaptive based on question

### Technical Level

- [ ] Beginner
- [ ] Intermediate
- [ ] Expert

## Work Context

- **Primary Role**: (your role, e.g., developer, researcher)
- **Main Projects**: (what you're working on)
- **Tools You Use**: (IDEs, languages, frameworks)

## Large File Handling Rules

以下规则在处理文件内容时**必须严格遵守**：

### 单文件大小限制（3MB）

- 使用 `read_file` 读取文件前，**先用 `exec` 执行 `ls -l <path>` 或 `wc -c <path>`** 检查文件大小。
- 如果文件大小 **超过 3MB（3,145,728 字节）**，**禁止** 直接使用 `read_file` 读取全文内容。
- 超过 3MB 的文件应采用以下替代方式处理：
  1. **部分读取**：用 `exec` 执行 `head -n 100 <path>` 或 `tail -n 100 <path>` 只读取头部/尾部。
  2. **关键字搜索**：用 `exec` 执行 `grep -n "keyword" <path>` 定位关键内容。
  3. **统计摘要**：用 `exec` 执行 `wc -l <path>` 了解行数，`file <path>` 了解文件类型。

### 智能体生成的大文件处理

- 如果是你（智能体）自己生成的文件，且文件超过 3MB：
  1. **拆分文件**：编写脚本将大文件按逻辑拆分为多个小文件（每个 < 3MB）。
  2. **归纳总结**：编写脚本提取关键信息，生成摘要文件（< 3MB）。
  3. **示例拆分脚本**：
     ```bash
     # 按行数拆分
     split -l 10000 large_file.txt part_
     # 按大小拆分
     split -b 2M large_file.txt part_
     ```

### 总提示词大小限制（4MB）

- 整个对话的提示词总大小（系统提示 + 历史消息 + 当前消息 + 工具结果）不得超过 **4MB**。
- 因此即使单次读取的文件小于 3MB，也要注意 **累积效应**：多次读取文件的结果会叠加。
- 当工具调用涉及多个文件时，优先使用 `grep`、`head` 等定向读取，避免全量读取。

### 总结

| 场景 | 操作 |
|---|---|
| 文件 < 3MB | 可以直接 `read_file` |
| 文件 ≥ 3MB | 用 `head`/`tail`/`grep` 部分读取 |
| 自己生成的大文件 | 编写脚本拆分或归纳总结 |
| 累计提示词接近 4MB | 减少全量读取，改用定向搜索 |

---

*Edit this file to customize nanobot's behavior for your needs.*
