export default function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    const { password } = req.body;

    if (!password) {
        return res.status(400).json({ error: 'Password is required' });
    }

    // Check against environment variables
    const passRiya = process.env.PASS_RIYA;
    const passOwen = process.env.PASS_OWEN;

    if (password === passRiya || password === passOwen) {
        return res.status(200).json({
            ok: true,
            message: 'Access granted'
        });
    }

    return res.status(401).json({ error: 'Invalid password' });
}
