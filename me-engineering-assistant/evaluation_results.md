# ME Engineering Assistant — Evaluation Results

**Date:** 2026-04-11
**Model:** gpt-4o-mini | Embedding: text-embedding-3-small
**Result:** 10/10 PASS (threshold: 8/10)

---

## Q1: What is the maximum operating temperature for the ECU-750?
**Category:** Single Source - ECU-700

The ECU-750 operates within a temperature range of -40°C to +85°C, making the maximum operating temperature +85°C. (source: ECU-700_Series_Manual.md)

---

## Q2: How much RAM does the ECU-850 have?
**Category:** Single Source - ECU-800

The ECU-850 has 2 GB of LPDDR4 RAM. (sources: ECU-800_Series_Base.md, ECU-800_Series_Plus.md)

---

## Q3: What are the AI capabilities of the ECU-850b?
**Category:** Single Source - ECU-800 Enhanced

The ECU-850b features advanced AI capabilities with a dedicated Neural Processing Unit (NPU) that provides 5 TOPS (Tera Operations Per Second) for handling edge AI workloads. This enhancement allows the ECU-850b to efficiently process complex algorithms and data for applications such as advanced driver-assistance systems (ADAS) and infotainment. (sources: ECU-800_Series_Base.md, ECU-800_Series_Plus.md)

---

## Q4: What are the differences between ECU-850 and ECU-850b?
**Category:** Comparative - Same Series

The ECU-850 features a dual-core ARM Cortex-A53 processor clocked at 1.2 GHz and comes with 2 GB of LPDDR4 RAM, while the ECU-850b has the same dual-core architecture but is clocked higher at 1.5 GHz and includes an additional dedicated Neural Processing Unit (NPU) capable of 5 TOPS for enhanced AI capabilities. Furthermore, the ECU-850b is equipped with 4 GB of LPDDR4 RAM and 32 GB of eMMC storage, compared to the ECU-850's 2 GB RAM and 16 GB storage. Overall, the ECU-850b offers significant improvements in processing power, memory, and AI functionality. (sources: ECU-800_Series_Base.md, ECU-800_Series_Plus.md)

---

## Q5: Compare the CAN bus capabilities of ECU-750 and ECU-850.
**Category:** Comparative - Cross Series

The ECU-750 features a single channel CAN FD interface with speeds up to 1 Mbps, which is suitable for basic automotive applications. In contrast, the ECU-850 is equipped with a dual channel CAN FD interface that supports speeds of up to 2 Mbps per channel, allowing for enhanced data throughput and redundancy in more complex automotive systems. Overall, the ECU-850 offers significantly improved CAN bus capabilities compared to the ECU-750, making it better suited for advanced applications. (sources: ECU-700_Series_Manual.md, ECU-800_Series_Base.md)

---

## Q6: What is the power consumption of the ECU-850b under load?
**Category:** Technical Specification

The power consumption of the ECU-850b under load is 1.7A. (sources: ECU-800_Series_Base.md, ECU-800_Series_Plus.md)

---

## Q7: Which ECU models support Over-the-Air (OTA) updates?
**Category:** Feature Availability

The ECU-750 does not support Over-the-Air (OTA) updates, as it is a legacy model. In contrast, both the ECU-850 and ECU-850b from the next-generation ECU-800 series do support OTA update capabilities, making them suitable for modern automotive applications that require software updates. The key takeaway is that OTA updates are available only in the ECU-800 series models. (sources: ECU-700_Series_Manual.md, ECU-800_Series_Base.md, ECU-800_Series_Plus.md)

---

## Q8: How does the storage capacity compare across all ECU models?
**Category:** Storage Comparison

The ECU-750 has a storage capacity of 2 MB of internal flash memory, while the ECU-850 features a significantly larger storage capacity of 16 GB eMMC. The ECU-850b further enhances this with 32 GB eMMC storage. This indicates a substantial increase in storage capabilities from the legacy ECU-750 to the next-gen ECU-800 series, allowing for more complex applications and data handling. (sources: ECU-700_Series_Manual.md, ECU-800_Series_Base.md, ECU-800_Series_Plus.md)

---

## Q9: Which ECU can operate in the harshest temperature conditions?
**Category:** Operating Environment

The ECU-750 operates within a temperature range of -40°C to +85°C, while both the ECU-850 and ECU-850b can function in a broader range of -40°C to +105°C. This means that the ECU-850 and ECU-850b can handle harsher temperature conditions compared to the ECU-750. Therefore, the ECU-850 and ECU-850b are better suited for extreme environments. (sources: ECU-700_Series_Manual.md, ECU-800_Series_Base.md, ECU-800_Series_Plus.md)

---

## Q10: How do you enable the NPU on the ECU-850b?
**Category:** Configuration/Usage

To enable the NPU on the ECU-850b, you can use the following driver command: `me-driver-ctl --enable-npu --mode=performance`. This command sets the NPU to performance mode, allowing it to operate effectively. (sources: ECU-800_Series_Base.md, ECU-800_Series_Plus.md)

---
