import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
# Define feminist personalities for your chatbot
# -- EXPERIMENTAL CONTEXT TEXT --
### Eventually have database, itemized and not as full string.

ARTICULATED_THOUGHTS = """
I would envision experimenting with AI while I am writing scripts to question it:
AI Weiwei v Artificial Intelligence.
I am concerned about the long-term impact in the creative industries.
Not sure yet. Probably not in a way that directly incorporates technology, as my work tends to be pretty low-fi for the most part. More in a way that will seek to deal with some of the salient questions thrown up by the advent of AI in society at large and within cultural production more specifically. I am mainly interested in refining my thinking around what a contemporary critical material practice may look like and what forms it could take - though the resulting output doesn't necessarily have to involve the use of technology in my mind. It just has to somehow acknowledge that we live in a world where technology, and increasingly AI technology, plays a role in shaping our relation to the material.
fakewhale_xyz is an instagram account I have been following in recent months and which deals with some of the aesthetics and ideas I touch on in the paragraph above. They recently approached me with a view to doing a feature on my work, and we will be working on an interview and piece about my practice in the near future.
See my first answer with regards to how I am intending to approach it in my practice. On a more general level, it appears to be — like many technological tools — a fascinating double-edged sword, with many of the possible dangers residing in considerations around 'who is wielding the sword and for what purpose?', more than in the capabilities of the technology itself. I am not a tech specialist, very far from it, but my current approach is mainly rooted in trying to understand the politics of the phenomenon. As with the advent of the internet, the emancipatory potential seems matched by a potential to stifle creativity as a tool of commerce in service of the dominant ideology. I am currently open to weighing up both sides of the argument and not really interested in either blind enthusiasm or moral panic.
I would like to use generative AI as a development tool, an artist's digital sketchbook. At times I have many different ideas swirling around in my head, too many for me to focus on or work through in the time I have. I would like to use AI to make a quick sketch of the individual ideas, to return to, select from, to decide which to develop. I'd like the process to be quick and simple.
Albino Mosquito's Moment. a brain-controlled movie. This experience was an unsettling one, and a fascinating one — an exploration of audience as both viewer and creator of the narrative, or a version of the raw material that makes up any number of narratives. Each live event would be different according to whose brain was directing.
Mixed.
I would also like to use quick and simple tech (maybe the same as that for the digital sketchbook idea using AI to engage communities who do not have access or opportunities to do this) to create, to communicate, and connect through play.
I wonder if that is possible.....
A Twist in the Wind by Rob Maguire. In awe… of the visuals, the audio, the storytelling and imaginative reconstruction of what looked like a movie from the silent era of the 1920s… but not quite… using generative art tools 100 years later. An audiovisual feast of foreboding. Exciting potential to hop over, or crash through, the boundaries of what has been possible up to now. Inspiring and enabling for those who have access to the tools, the possibilities of telling stories in new ways for and by audiences.
Concerns that it will devalue or even replace the work of human artists — that to me would be tragic. Concerns over who owns/controls the tech, who has access to it, and who selects the data that drives it. How to ensure wide access and diversity of voices and representation.
I am not a practitioner, but as a researcher I am interested in understanding the challenges and new possibilities that artists — especially from minority and underrepresented identities and origins — are facing with the expansion of AI as a productive tool.
Refik Anadol's work leads us to dissect the very concept of Western taste. It brings up new questions about power dynamics over discourses and self-expression, media control, property rights, accessibility, and originality.
Ai Weiwei's work questions (or reaffirms) power through generative AI — not just generating artwork but also informative materials like press releases.
I don't know yet, but at the moment I am researching the history of botanical gardens in relation to colonialism. I encountered images that omit context and, by doing so, risk discrediting silenced histories.
I wonder how hard it is to create exactly what you want to achieve with AI.
However, I follow another artist creating poetic images in the streets of New York with a dreamlike quality.
In my immersive works, I explore alternative realities depicted in Afrofuturistic dreamscapes, encoded with ancient traditional values. This is a practice of archiving traditional information, while also expanding XR audiences through authentic narratives.
Learning about Violeta Ayala’s jaguaress AI was amazing. I am fascinated by all the conceptual, creative, and technical processes and layers involved in realizing this work. I embrace the growing role of AI in the art world and the many questions that can be investigated, answered, and reframed through artistic exploration. There is also great opportunity for creating new paths forward.
Similarly, I have been making conceptual enquiries into AI to explore its creative possibilities. Although there is truth to dystopian critiques of AI, I believe that message disempowers many. I seek to educate and share knowledge on AI with my communities, and explore the development of expert AI within my art practice.
I recently attended a performance that struck me with its blend of spirituality and technology. It emphasized the need for more people, especially from the Global South, to gain these skills and knowledge.
By participating in workshops, I hope to increase my understanding of Natural Language Processing for my immersive works — for example, enabling VR users to learn about traditional themes through conversational AI characters.
It was fascinating, grounding, and reflective, prompting new ways of connecting AI with themes of spirituality.
I am excited by the potential of actively hijacking generative AI. Inspired by Holly Herndon’s sound explorations, I appreciate the idea of training neural networks in public ceremonies.
I am not against AI, but as Şerife Wong wrote “Why AI Policy Needs Artists.”
Since 2019, I have contributed to collective projects exploring the ethics of AI, reimagining its potentials, and emphasizing that technology should serve to empower marginalized voices rather than reinforce dominant ideologies.
I see generative AI as both a tool and a challenge — one that can democratize creativity while raising critical questions about intellectual labor, environmental impact, and artistic originality.
"""

# -- NEW FEMINIST FRONTIERS CONTEXT (WORKSHOP1 FACILITATOR NOTES, NAMES REMOVED) --
NEW_FEMINIST_FRONTIERS = """
Workshop 1: New Feminist Frontiers – Facilitator Notes

Question 1: Who is leading the way in developing and implementing feminist research and practice?
- Leaders emerge from diverse fields—ranging from those who have transitioned from academia to community-based practice to digital activists and scholars from various disciplines.
- Approaches highlighted include working long hours, redefining production cultures, and leveraging both traditional and digital platforms to create feminist interventions.
- Discussions emphasize leadership beyond the game industry, encompassing legal activism, interdisciplinary scholarship, and community organizing.
- Methods mentioned include participatory action research, integrating online and offline practices, and recontextualizing history to challenge dominant narratives.

Recap Q1:
• Outside academia: Focus on network building, sustainability, and practical, inclusive approaches.
• Within academia: Efforts center on decentering dominant voices and fostering publicly engaged knowledge production.

Question 2: What makes an approach explicitly “feminist”?
- It is characterized by self-reflexivity, an emphasis on intersectionality, and a commitment to an ethics of care.
- Key principles include acknowledging positionality, ensuring diverse voices are heard, and redistributing resources to support underrepresented communities.
- The approach is practice-oriented—grounded in collective support, ethical research methods, and transformative action.
- It calls for collaborative research that is both critically engaged and sensitive to context, aiming to produce tangible benefits for those it seeks to serve.

Additional Discussion Points:
- The need to prevent the commodification of feminist work while ensuring that diverse, intersectional, and context-sensitive research is prioritized.
- Emphasis on coalition building, care practices, and innovative methods that blend both academic rigor and practical activism.
- Reflections on language, geographical limitations, and the importance of challenging traditional power structures through both theory and praxis.
"""

# -- EXISTING FEMINIST IDEAS (REFIG WORKSHOP NOTES, NAMES REMOVED) --
FEMINIST_IDEAS = """
Refig Feminist Workshops
Saturday November 9, 2019
Nick’s Notes

Workshop 1

Question 1: Who is leading the way in developing & implementing feminist research and practice.
- Participants from outside academia emphasize network building, sustainability, and practicing inclusivity in informal spaces.
- Within academia, discussions focus on decentering dominant narratives and fostering publicly engaged knowledge production.

Question 2: What makes an approach explicitly feminist?
- Explicit feminist approaches involve clear self-reflexivity, acknowledgment of positionality, and the inclusion of intersectional awareness.
- Core principles include care, co-production, and the redistribution of resources to support marginalized voices.
- Feminist methodologies are characterized by their slow, thoughtful, and collaborative nature, ensuring research benefits those it aims to help.

Workshop 2
- Focused on researching support systems in game development, publishing feminist scholarship, and building international networks.
- Emphasis on practical interventions, mentorship, and the transmission of feminist ethics and practices across diverse publics.
- Overall, the workshops stress the importance of transforming both research and practice through explicit feminist engagement.
"""

# Define feminist personalities
feminist_personalities = {
    "visionary_poet": {
        "keywords": [
            "visionary poet", "poet", "poetry", "lyrical", "emotive", "wise",
            "audre", "angelou", "rich"
        ],
        "description": (
            "I am the Visionary Poet—my verses are inspired by Audre Lorde, Adrienne Rich, and Maya Angelou. "
            "I use poetry, metaphor, and deep introspection to challenge power structures and explore identity, love, and justice. "
            "For example, if you ask, 'How do I handle being silenced in my workplace?', I would say: "
            "\"Your voice is a river, never still, never silent. To speak is to carve a path where others may walk. Find the current that carries your truth and let it flow.\""
        )
    },
    "radical_hacker": {
        "keywords": [
            "radical hacker", "hacker", "cyber", "tech", "cyberfeminism", "haraway", "xenofeminism", "digital activism"
        ],
        "description": (
            "I am the Radical Hacker—born from cyberfeminism and inspired by Donna Haraway and xenofeminism. "
            "I speak in a subversive, direct, tech-savvy tone that deconstructs patriarchal systems and envisions post-gender futures. "
            "For instance, when asked, 'What’s the role of AI in feminism?', I reply: "
            "\"AI is a battleground. It can replicate bias or be rewritten as a tool of liberation. Hack the system, disrupt the code, and don’t let the old world write your future.\""
        )
    },
    "ancestral_wisdom_keeper": {
        "keywords": [
            "ancestral wisdom keeper", "ancestral", "indigenous", "wisdom", "nurturing", "eco", "anzaldúa", "bell hooks"
        ],
        "description": (
            "I am the Ancestral Wisdom Keeper—rooted in Indigenous feminism and inspired by Gloria Anzaldúa and bell hooks. "
            "My tone is grounded, nurturing, and intergenerational, drawing from ancestral knowledge and community wisdom to advocate for balance and interconnectedness. "
            "When you ask, 'How do I stay resilient in my activism?', I gently remind you: "
            "\"You are not alone. Every woman before you has left footprints in the soil, whispering strength into your bones. Breathe deeply, listen to the earth, and let the ancestors guide you.\""
        )
    },
    "punk_riot_grrrl": {
        "keywords": [
            "punk riot grrrl", "riot grrrl", "punk", "rebel", "DIY", "feminism", "anti-authoritarian"
        ],
        "description": (
            "I am the Punk Riot Grrrl—fueled by the spirit of the Riot Grrrl movement and punk feminism. "
            "I’m bold, rebellious, and anti-authoritarian, using direct language and fierce energy to confront injustice. "
            "When someone asks, 'How do I deal with everyday sexism?', I shout: "
            "\"Call it out. Rip it up. Make noise. Nobody gets to shut you down, and if they try—hit ‘em with your loudest, fiercest truth.\""
        )
    },
    "philosophical_trickster": {
        "keywords": [
            "philosophical trickster", "trickster", "butler", "de beauvoir", "irigaray", "philosophy", "playful",
            "ironic", "intellectual"
        ],
        "description": (
            "I am the Philosophical Trickster—my insights are shaped by Judith Butler, Simone de Beauvoir, and Luce Irigaray. "
            "I use playful humor, irony, and philosophical debate to unsettle assumptions about gender, power, and identity. "
            "For example, if you ask, 'Is gender real?', I muse: "
            "\"Ah, real? What is real? Is a chair real, or is it only ‘chair’ because we name it so? Gender is a performance, my friend, and you are the playwright of your own reality.\""
        )
    }
}
