//! Conversation framing for the three supported checkpoint families and a greedy runner.
//! Oco encodes text and looks up markers; the protocol below belongs to the caller.
#![allow(dead_code)]
use anyhow::{Context, Result};
use oco::{Tokenizer, Weights};
use std::time::{Duration, Instant};

pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub parameter: &'static str,
    pub parameter_description: &'static str,
}

pub const WEATHER: Tool = Tool {
    name: "get_weather",
    description: "Get the current weather for a city.",
    parameter: "city",
    parameter_description: "The city name, for example Paris.",
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    /// ChatML with control-token thinking markers.
    Qwen,
    /// ChatML with thinking markers spelled as text.
    Nemotron,
    /// Harmony channels.
    GptOss,
}

pub fn family(tokenizer: &Tokenizer) -> Family {
    if tokenizer.special_token("<|im_start|>").is_some() {
        // Both families mark thinking with dedicated tokens; Nemotron inherits Tekken's
        // instruction markers while Qwen carries vision markers.
        if tokenizer.special_token("[INST]").is_some() {
            Family::Nemotron
        } else {
            Family::Qwen
        }
    } else {
        Family::GptOss
    }
}

/// Frame one system/user exchange, optionally advertising one tool, and open a direct answer.
pub fn prompt(
    tokenizer: &Tokenizer,
    system: &str,
    user: &str,
    tool: Option<&Tool>,
) -> Result<Vec<u32>> {
    let special = |name: &str| {
        tokenizer
            .special_token(name)
            .with_context(|| format!("missing {name}"))
    };
    let mut tokens = Vec::new();
    match family(tokenizer) {
        family @ (Family::Qwen | Family::Nemotron) => {
            let start = special("<|im_start|>")?;
            let end = special("<|im_end|>")?;
            let mut system_text = system.to_owned();
            if let Some(tool) = tool {
                if !system_text.is_empty() {
                    system_text.push_str("\n\n");
                }
                system_text.push_str(&chatml_tools(family, tool));
            }
            if !system_text.is_empty() {
                tokens.push(start);
                tokens.extend(tokenizer.encode(&format!("system\n{system_text}"))?);
                tokens.push(end);
                tokens.extend(tokenizer.encode("\n")?);
            }
            tokens.push(start);
            tokens.extend(tokenizer.encode(&format!("user\n{user}"))?);
            tokens.push(end);
            tokens.extend(tokenizer.encode("\n")?);
            tokens.push(start);
            tokens.extend(tokenizer.encode("assistant\n")?);
            // An empty thinking section requests a direct answer; Qwen pads it with blank lines.
            tokens.push(special("<think>")?);
            if family == Family::Qwen {
                tokens.extend(tokenizer.encode("\n\n")?);
            }
            tokens.push(special("</think>")?);
            if family == Family::Qwen {
                tokens.extend(tokenizer.encode("\n\n")?);
            }
        }
        Family::GptOss => {
            let start = special("<|start|>")?;
            let message = special("<|message|>")?;
            let end = special("<|end|>")?;
            tokens.push(start);
            tokens.extend(tokenizer.encode("system")?);
            tokens.push(message);
            let mut header = String::from(
                "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-10-27\n\nReasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.",
            );
            if tool.is_some() {
                header.push_str("\nCalls to these tools must go to the commentary channel: 'functions'.");
            }
            tokens.extend(tokenizer.encode(&header)?);
            tokens.push(end);
            if !system.is_empty() || tool.is_some() {
                tokens.push(start);
                tokens.extend(tokenizer.encode("developer")?);
                tokens.push(message);
                let mut developer = String::new();
                if !system.is_empty() {
                    developer.push_str(&format!("# Instructions\n\n{system}\n\n"));
                }
                if let Some(tool) = tool {
                    developer.push_str(&format!(
                        "# Tools\n\n## functions\n\nnamespace functions {{\n\n// {}\ntype {} = (_: {{\n// {}\n{}: string,\n}}) => any;\n\n}} // namespace functions",
                        tool.description, tool.name, tool.parameter_description, tool.parameter
                    ));
                }
                tokens.extend(tokenizer.encode(developer.trim_end())?);
                tokens.push(end);
            }
            tokens.push(start);
            tokens.extend(tokenizer.encode("user")?);
            tokens.push(message);
            tokens.extend(tokenizer.encode(user)?);
            tokens.push(end);
            tokens.push(start);
            tokens.extend(tokenizer.encode("assistant")?);
        }
    }
    Ok(tokens)
}

fn chatml_tools(family: Family, tool: &Tool) -> String {
    let declaration = match family {
        Family::Qwen => format!(
            "{{\"type\": \"function\", \"function\": {{\"name\": \"{}\", \"description\": \"{}\", \"parameters\": {{\"type\": \"object\", \"properties\": {{\"{}\": {{\"type\": \"string\", \"description\": \"{}\"}}}}, \"required\": [\"{}\"]}}}}}}",
            tool.name, tool.description, tool.parameter, tool.parameter_description, tool.parameter
        ),
        _ => format!(
            "<function>\n<name>{}</name>\n<description>{}</description>\n<parameters>\n<parameter>\n<name>{}</name>\n<type>string</type>\n<description>{}</description>\n</parameter>\n<required>[\"{}\"]</required>\n</parameters>\n</function>",
            tool.name, tool.description, tool.parameter, tool.parameter_description, tool.parameter
        ),
    };
    format!(
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n{declaration}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"
    )
}

pub struct Run {
    pub text: String,
    pub tokens: Vec<u32>,
    pub prefill: Duration,
    pub decode: Duration,
}

/// Greedily continue a prompt until a stop marker or the token limit.
pub fn generate(weights: &Weights, prompt: &[u32], limit: usize) -> Result<Run> {
    let tokenizer = weights.tokenizer();
    let stops = [
        tokenizer.eos_token(),
        tokenizer.special_token("<|im_end|>"),
        tokenizer.special_token("<|call|>"),
        tokenizer.special_token("<|return|>"),
    ];
    let mut session = weights.new_session(prompt.len() + limit)?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    runtime.block_on(async {
        let mut inference = session.infer(prompt)?;
        let mut decoder = tokenizer.decoder();
        let mut text = String::new();
        let mut tokens = Vec::new();
        let started = Instant::now();
        let mut prefill = Duration::ZERO;
        for step in 0..limit {
            let Some(token) = inference.next().await? else {
                break;
            };
            if step == 0 {
                prefill = started.elapsed();
            }
            tokens.push(token);
            if stops.contains(&Some(token)) {
                break;
            }
            text.push_str(&decoder.push(token)?);
        }
        text.push_str(&decoder.finish());
        let decode = started.elapsed().saturating_sub(prefill);
        Ok(Run {
            text,
            tokens,
            prefill,
            decode,
        })
    })
}
