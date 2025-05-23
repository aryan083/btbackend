from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn
from rank_bm25 import BM25Okapi

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Comprehensive text cleaning pipeline"""
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Text cleaning
    cleaned = []
    for token in tokens:
        # Remove special characters and numbers
        token = re.sub(r'[^a-zA-Z]', '', token)
        if len(token) < 2:
            continue
            
        # Stopword removal
        if token in stop_words:
            continue
            
        # Stemming
        cleaned.append(stemmer.stem(token))
    
    return cleaned
def process_article(article_text):
    """Enhanced article processing with NLP pipeline"""
    chunks = []
    current_section = []
    
    # Structural processing
    for line in article_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Section detection
        if re.match(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)*(?: - \d+\.\d+)?$', line):
            if current_section:
                chunks.append(' '.join(current_section))
                current_section = []
            continue
                
        # Content chunking
        sentences = nltk.sent_tokenize(line)
        for sent in sentences:
            if len(current_section) < 3:
                current_section.append(sent)
            else:
                chunks.append(' '.join(current_section))
                current_section = [sent]
    
    if current_section:
        chunks.append(' '.join(current_section))
    
    # Create search corpus with full preprocessing
    return {
        'original': chunks,
        'processed': [preprocess_text(chunk) for chunk in chunks]
    }
def answer_question(question, article_data):
    """Enhanced QA with full text normalization"""
    # Process question
    query_terms = preprocess_text(question)

    # BM25 search
    bm25 = BM25Okapi(article_data['processed'])
    bm25_scores = bm25.get_scores(query_terms)

    # Semantic search
    question_embed = model.encode([question])
    chunk_embeds = model.encode(article_data['original'])
    semantic_scores = np.dot(chunk_embeds, question_embed.T).flatten()

    # Combine scores with normalization
    combined = []
    # Check if bm25_scores is not empty before finding max
    max_bm25 = max(bm25_scores) if bm25_scores.size > 0 and max(bm25_scores) > 0 else 1
    # Check if semantic_scores is not empty before finding max
    max_semantic = max(semantic_scores) if semantic_scores.size > 0 and max(semantic_scores) > 0 else 1

    for b, s in zip(bm25_scores, semantic_scores):
        combined.append(
            (0.6 * (b/max_bm25)) + (0.4 * (s/max_semantic))
        )

    # Get best match
    # Handle case where combined is empty
    if not combined:
        return "Could not find a relevant answer."

    best_idx = np.argmax(combined)
    return article_data['original'][best_idx]

ARTICLE = """
congestion control - 5.3
Congestion control is a critical aspect of network engineering, ensuring fair and efficient use of network resources. It aims to prevent or alleviate network congestion, a state where the network's load exceeds its capacity, leading to increased delays, packet loss, and reduced throughput. This document provides an exhaustive overview of congestion control mechanisms, encompassing definitions, technical breakdowns, practical examples, common pitfalls, and advanced usage patterns.

Comprehensive Definition of Congestion Control
Congestion control encompasses the mechanisms and techniques used to manage traffic flow within a network. Its primary goal is to prevent or minimize the adverse effects of congestion, thereby maintaining acceptable levels of network performance. This involves regulating the rate at which sources inject traffic into the network, ensuring that the aggregate demand does not overwhelm the available resources. Congestion, in essence, arises when the number of packets sent to the network exceeds its ability to process them efficiently, which translates to an undesirable experience for end users.

Formally, congestion control is the process of regulating the amount of data entering a network to prevent overload and maintain acceptable levels of performance, including throughput and delay. It achieves this by monitoring network conditions, detecting congestion, and then adjusting the sending rate of data sources.

Detailed Technical Breakdown of Components
Congestion control mechanisms can be broadly categorized into two main approaches:

Open-Loop Congestion Control (Prevention): This approach aims to prevent congestion before it occurs. It relies on pre-emptive measures taken by the source or destination, without explicit feedback from the network.
Closed-Loop Congestion Control (Removal): This approach focuses on alleviating congestion *after* it has occurred. It involves monitoring network conditions, detecting congestion, and then adjusting the sending rates of data sources based on this feedback.
Open-Loop Congestion Control Techniques
Retransmission Policy: A well-designed retransmission policy is crucial to avoid exacerbating congestion. Aggressive retransmission timers can lead to unnecessary packet duplication, increasing the load on the network. Techniques like exponential backoff are employed to reduce the retransmission rate during periods of congestion.
Window Policy: The type of windowing mechanism used can significantly affect congestion. Selective Repeat ARQ is generally superior to Go-Back-N ARQ in congested networks. Go-Back-N can lead to the retransmission of multiple packets, even if some have been successfully received, thus worsening congestion.
Acknowledgment Policy: The frequency of acknowledgments can impact network load. Sending fewer acknowledgments reduces the overhead on the network. Techniques like delayed acknowledgments, where acknowledgments are sent only if there is data to send or after a certain timer expires, can be employed.
Discarding Policy: Routers can employ intelligent discarding policies to minimize the impact of packet loss. For example, in layered encoded video streams, less important layers can be discarded first.
Admission Control: In virtual-circuit networks, admission control can prevent congestion by denying new connections if the network is already congested or likely to become congested in the future. This is a Quality of Service (QoS) mechanism.
Closed-Loop Congestion Control Techniques
Backpressure: A congested node stops receiving data from its upstream nodes, propagating the congestion signal back towards the source. This is typically used in virtual-circuit networks where each node knows its upstream neighbor. Backpressure isn't usable in datagram networks because routers don't know where packets came from.
Choke Packet: A congested node sends a choke packet directly to the source, informing it of the congestion. The source then reduces its sending rate. ICMP Source Quench messages are an example of a choke packet mechanism, though they are rarely used now.
Implicit Signaling: The source infers congestion based on indirect cues, such as packet loss or increased round-trip time (RTT). TCP's congestion control mechanisms rely heavily on implicit signaling.
Explicit Signaling: The network explicitly signals congestion to the source or destination. This can be done through:
Backward Signaling: A bit is set in packets traveling in the reverse direction of data flow (e.g., BECN in Frame Relay).
Forward Signaling: A bit is set in packets traveling in the direction of data flow (e.g., FECN in Frame Relay).
TCP Congestion Control
TCP employs a sophisticated set of mechanisms to manage congestion:

Congestion Window (cwnd): A variable that limits the amount of data a sender can have in flight, in addition to the receiver advertised window (rwnd). The actual window size is the minimum of cwnd and rwnd: window_size = min(cwnd, rwnd).
Slow Start: The sender starts with a small cwnd and increases it exponentially until a threshold (ssthresh) is reached. For each ACK received, the cwnd increases by one Maximum Segment Size (MSS).
Congestion Avoidance: Once ssthresh is reached, the sender switches to additive increase. For each Round Trip Time (RTT), the cwnd increases by one MSS.
Congestion Detection: When congestion is detected (either through a timeout or the reception of three duplicate ACKs), the sender reduces its sending rate.
Timeout: ssthresh is set to half of the current cwnd, and cwnd is reset to one MSS. Slow start is then initiated.
Three Duplicate ACKs (Fast Retransmit/Fast Recovery): ssthresh is set to half of the current cwnd, and cwnd is set to the new ssthresh. Congestion avoidance is then initiated. Some implementations will add three segment sizes to the ssthresh.
Frame Relay Congestion Control
Frame Relay employs explicit congestion notification:

Backward Explicit Congestion Notification (BECN): A bit set in frames traveling in the reverse direction to inform the source of congestion.
Forward Explicit Congestion Notification (FECN): A bit set in frames traveling in the forward direction to inform the destination of congestion. The destination can then signal the source (e.g., by delaying acknowledgments).
Numerous Practical Examples
Example 1: TCP Slow Start
A TCP connection begins with cwnd = 1 MSS (e.g., 1460 bytes). The sender transmits one segment. Upon receiving an ACK, cwnd increases to 2 MSS (2920 bytes). The sender transmits two segments. Upon receiving both ACKs, cwnd increases to 4 MSS (5840 bytes), and so on. This exponential growth continues until ssthresh is reached or congestion is detected.

 Initial cwnd = 1460 bytes After 1st ACK: cwnd = 2920 bytes After 2nd ACK: cwnd = 5840 bytes ... (exponential increase) 
Example 2: TCP Congestion Avoidance
Assume ssthresh = 10 MSS (14600 bytes) has been reached. The sender transmits 10 segments. Upon receiving all 10 ACKs (after one RTT), cwnd increases by one MSS to 11 MSS (16060 bytes). This additive increase continues until congestion is detected.

 ssthresh = 14600 bytes cwnd = 14600 bytes After 1 RTT: cwnd = 16060 bytes ... (linear increase) 
Example 3: TCP Timeout Event
cwnd = 20 MSS, ssthresh = 15 MSS. A timeout occurs. ssthresh is set to cwnd/2 = 10 MSS. cwnd is reset to 1 MSS. Slow start is initiated.

 cwnd = 20 MSS ssthresh = 15 MSS Timeout Occurs New ssthresh = 10 MSS New cwnd = 1 MSS Slow Start initiated 
Example 4: Frame Relay BECN
A switch in a Frame Relay network experiences congestion. It sets the BECN bit in frames traveling towards the source. The source, upon receiving a frame with BECN set, reduces its sending rate to alleviate congestion.

Example 5: Frame Relay FECN
A switch in a Frame Relay network experiences congestion. It sets the FECN bit in frames traveling towards the destination. The destination, upon receiving a frame with FECN set, delays acknowledgments to slow down the source.

Common Pitfalls and Troubleshooting
Aggressive Retransmission Timers: Setting retransmission timers too low can lead to unnecessary retransmissions, exacerbating congestion.
Ignoring Explicit Congestion Signals: Failing to respond to BECN or FECN signals can lead to further packet loss and degraded performance.
Incorrect ssthresh Values: Improperly configured ssthresh values in TCP can lead to either overly conservative or overly aggressive sending behavior.
Bufferbloat: Excessively large buffers in network devices can mask congestion, leading to increased latency and potentially even more severe congestion when the buffers eventually overflow. Modern AQM (Active Queue Management) techniques seek to alleviate this.
Starvation due to Priority Queuing: Unfair or badly configured priority queues can lead to starvation of lower priority traffic.
Troubleshooting Congestion Issues
Monitor Network Performance: Use tools to monitor packet loss, latency, and throughput to identify congestion points.
Analyze Traffic Patterns: Identify sources and destinations contributing to congestion.
Adjust TCP Parameters: Tune TCP parameters such as initial window size, ssthresh, and retransmission timers.
Implement AQM: Deploy Active Queue Management techniques like RED (Random Early Detection) or CoDel (Controlled Delay) to proactively manage queues and prevent bufferbloat.
Apply Traffic Shaping: Use traffic shaping techniques to smooth out bursty traffic and prevent it from overwhelming the network.
Implement QoS Mechanisms: Use QoS mechanisms to prioritize critical traffic and ensure that it receives adequate bandwidth.
Advanced Usage Patterns
Explicit Congestion Notification (ECN): An extension to IP that allows routers to explicitly signal congestion without dropping packets. The source and destination must both support ECN.
Data Center TCP (DCTCP): A modified version of TCP designed for low-latency, high-bandwidth data center networks. DCTCP uses ECN to provide more precise congestion feedback, enabling faster adaptation to changing network conditions.
Proportional Rate Reduction (PRR): An algorithm that improves TCP's response to congestion by more accurately reducing the sending rate based on the amount of data lost.
TCP BBR (Bottleneck Bandwidth and RTT): A congestion control algorithm developed by Google that attempts to estimate the bottleneck bandwidth and RTT of the network path, and then uses this information to optimize the sending rate. BBR is designed to achieve high throughput and low latency, even in challenging network conditions.
Cross-References to Related Concepts
Flow Control: Mechanisms to prevent a sender from overwhelming a receiver. While flow control operates end-to-end between a sender and receiver, congestion control addresses network-wide congestion issues.
Quality of Service (QoS): Mechanisms to provide different levels of service to different traffic flows. Congestion control is often a component of a broader QoS strategy.
Active Queue Management (AQM): Techniques used by routers to manage queues and prevent bufferbloat, often in conjunction with congestion control mechanisms.
Network Layer: Congestion control is implemented within network and transport layers.
Transport Layer: TCP is the main protocol to achieve congestion control.
Comparative Analysis of Congestion Control Techniques
Technique	Category	Advantages	Disadvantages	Applicability
Slow Start	TCP Congestion Control	Rapidly increases sending rate when network is uncongested.	Can lead to congestion if initial estimate is too high.	All TCP connections.
Congestion Avoidance	TCP Congestion Control	Linearly increases sending rate, avoiding sudden bursts.	Slower to adapt to available bandwidth than slow start.	TCP connections after reaching ssthresh.
Fast Retransmit/Fast Recovery	TCP Congestion Control	Quickly recovers from packet loss without waiting for a timeout.	Can be triggered by out-of-order packets, leading to unnecessary rate reduction.	TCP connections.
BECN	Frame Relay Congestion Control	Provides explicit congestion feedback to the source.	Requires support from both the network and the source.	Frame Relay networks.
FECN	Frame Relay Congestion Control	Provides congestion feedback to the destination, allowing it to signal the source.	Less direct than BECN, relying on the destination to signal the source.	Frame Relay networks.
Traffic Descriptors and Traffic Profiles
Understanding the characteristics of data traffic is crucial for effective congestion control and quality of service. Traffic descriptors provide qualitative values that represent a data flow, enabling network devices to make informed decisions about resource allocation and traffic management.

Traffic Descriptors
Key traffic descriptors include:

Average Data Rate: The number of bits sent during a period of time, divided by the duration of that period. It represents the average bandwidth needed by the traffic flow.
Peak Data Rate: The maximum data rate of the traffic flow. It indicates the peak bandwidth that the network needs to accommodate the traffic without altering its flow.
Maximum Burst Size: The maximum length of time the traffic is generated at the peak rate. This is important because short bursts can often be tolerated, while longer bursts may cause congestion.
Effective Bandwidth: The bandwidth that the network needs to allocate for the flow of traffic. This is a complex calculation dependent on the average data rate, peak data rate, and maximum burst size.
Traffic Profiles
Data flows can exhibit different traffic profiles:

Constant Bit Rate (CBR): The data rate does not change over time. The average data rate and peak data rate are the same. The maximum burst size is not applicable. This type of traffic is very predictable and easy to handle. Example application: uncompressed audio or video streaming.
Variable Bit Rate (VBR): The data rate changes over time, but the changes are relatively smooth. The average data rate and the peak data rate are different. The maximum burst size is usually small. This traffic requires more sophisticated handling than CBR. Example application: compressed video conferencing.
Bursty: The data rate changes suddenly in a very short time. The average bit rate and peak bit rate are very different, and the maximum burst size is significant. This is the most difficult type of traffic to handle, as it is unpredictable. Example application: web browsing, file transfer.
"""
processed_article = process_article(ARTICLE)
questions = [
    "What is prr?"
]
for q in questions:
    print(f"Q: {q}")
    print(f"A: {answer_question(q, processed_article)}\n")