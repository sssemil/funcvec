pub mod parsing {
    /// funcvec:group=parse_number
    pub fn parse_count(input: &str) -> Option<u32> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return None;
        }
        trimmed.parse::<u32>().ok()
    }

    /// funcvec:group=parse_number
    pub fn parse_quantity(raw: &str) -> Option<u32> {
        let value = raw.trim();
        if value.is_empty() {
            return None;
        }
        value.parse::<u32>().ok()
    }

    pub fn parse_boolish(input: &str) -> Option<bool> {
        let trimmed = input.trim().to_ascii_lowercase();
        match trimmed.as_str() {
            "true" | "yes" | "1" => Some(true),
            "false" | "no" | "0" => Some(false),
            _ => None,
        }
    }
}

pub mod totals {
    /// funcvec:group=sum_positive
    pub fn sum_positive(items: &[i32]) -> i32 {
        let mut total = 0;
        for item in items {
            if *item > 0 {
                total += *item;
            }
        }
        total
    }

    /// funcvec:group=sum_positive
    pub fn add_positive_values(values: &[i32]) -> i32 {
        values
            .iter()
            .filter(|value| **value > 0)
            .fold(0, |acc, value| acc + value)
    }

    pub fn sum_negative(items: &[i32]) -> i32 {
        let mut total = 0;
        for item in items {
            if *item < 0 {
                total += *item;
            }
        }
        total
    }
}

pub mod text {
    /// funcvec:group=slugify
    pub fn slugify_title(title: &str) -> String {
        title
            .trim()
            .to_ascii_lowercase()
            .chars()
            .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
            .collect::<String>()
            .split('-')
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("-")
    }

    /// funcvec:group=slugify
    pub fn normalize_heading(heading: &str) -> String {
        heading
            .trim()
            .to_ascii_lowercase()
            .chars()
            .map(|item| if item.is_ascii_alphanumeric() { item } else { '-' })
            .collect::<String>()
            .split('-')
            .filter(|segment| !segment.is_empty())
            .collect::<Vec<_>>()
            .join("-")
    }

    pub fn initials(name: &str) -> String {
        name.split_whitespace()
            .filter_map(|part| part.chars().next())
            .collect::<String>()
            .to_ascii_uppercase()
    }
}

pub mod validation {
    /// funcvec:group=emailish
    pub fn looks_like_email(input: &str) -> bool {
        let trimmed = input.trim();
        if trimmed.is_empty() || trimmed.contains(' ') {
            return false;
        }
        let mut parts = trimmed.split('@');
        let Some(local) = parts.next() else {
            return false;
        };
        let Some(domain) = parts.next() else {
            return false;
        };
        parts.next().is_none() && !local.is_empty() && domain.contains('.')
    }

    /// funcvec:group=emailish
    pub fn is_probable_email(value: &str) -> bool {
        let candidate = value.trim();
        if candidate.is_empty() || candidate.contains(' ') {
            return false;
        }
        let mut chunks = candidate.split('@');
        let Some(name) = chunks.next() else {
            return false;
        };
        let Some(host) = chunks.next() else {
            return false;
        };
        chunks.next().is_none() && !name.is_empty() && host.contains('.')
    }

    pub fn looks_like_phone(input: &str) -> bool {
        let digits = input.chars().filter(|ch| ch.is_ascii_digit()).count();
        digits >= 10 && input.chars().all(|ch| ch.is_ascii_digit() || " -()+".contains(ch))
    }
}

pub mod records {
    #[derive(Clone, Debug)]
    pub struct Event {
        pub kind: String,
        pub count: u32,
        pub active: bool,
    }

    /// funcvec:group=active_counts
    pub fn active_total(events: &[Event]) -> u32 {
        let mut total = 0;
        for event in events {
            if event.active {
                total += event.count;
            }
        }
        total
    }

    /// funcvec:group=active_counts
    pub fn sum_enabled(records: &[Event]) -> u32 {
        records
            .iter()
            .filter(|record| record.active)
            .map(|record| record.count)
            .sum()
    }

    pub fn inactive_total(events: &[Event]) -> u32 {
        let mut total = 0;
        for event in events {
            if !event.active {
                total += event.count;
            }
        }
        total
    }
}

pub mod windows {
    /// funcvec:group=moving_average
    pub fn moving_average(values: &[f64], width: usize) -> Vec<f64> {
        if width == 0 || values.len() < width {
            return Vec::new();
        }
        let mut out = Vec::new();
        for window in values.windows(width) {
            let sum: f64 = window.iter().sum();
            out.push(sum / width as f64);
        }
        out
    }

    /// funcvec:group=moving_average
    pub fn rolling_mean(samples: &[f64], size: usize) -> Vec<f64> {
        if size == 0 || samples.len() < size {
            return Vec::new();
        }
        samples
            .windows(size)
            .map(|chunk| chunk.iter().sum::<f64>() / size as f64)
            .collect()
    }

    pub fn rolling_max(samples: &[f64], size: usize) -> Vec<f64> {
        if size == 0 || samples.len() < size {
            return Vec::new();
        }
        samples
            .windows(size)
            .map(|chunk| chunk.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            .collect()
    }
}

pub struct Account {
    balance: i64,
}

impl Account {
    pub fn balance(&self) -> i64 {
        self.balance
    }

    /// funcvec:group=credit
    pub fn deposit(&mut self, cents: i64) {
        if cents <= 0 {
            return;
        }
        self.balance += cents;
    }

    /// funcvec:group=credit
    pub fn add_credit(&mut self, amount: i64) {
        if amount <= 0 {
            return;
        }
        self.balance += amount;
    }
}

#[cfg(test)]
mod tests {
    /// funcvec:group=test_helpers
    fn make_name(id: u32) -> String {
        format!("user-{id}")
    }

    /// funcvec:group=test_helpers
    fn build_name(value: u32) -> String {
        format!("user-{value}")
    }

    #[test]
    fn smoke() {
        assert_eq!(make_name(7), build_name(7));
    }
}
